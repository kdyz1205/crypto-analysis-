"""End-to-end inference service.

Boots from a manifest. Caller pushes closed bars into the FeatureCache;
calls predict(symbol, tf) when a higher-tf candle closes; gets back a
PredictionRecord with versions, confidence, and the predicted next-token
+ behaviour distributions.

Strict contract:
- The manifest's tokenizer_version must equal the RuntimeTokenizer's.
- Predictions only fire when there are >= MIN_BARS bars in the cache.
- The price window comes from FeatureCache (training-parity feature builder).
- The trendline records used as input come from the runtime detector
  on the same DataFrame (no future bars).
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..models.config import FusionConfig
from ..models.full_model import TrendlineFusionModel
from ..registry.manifest import ArtifactManifest
from .feature_cache import FeatureCache
from .runtime_detector import detect_lines
from .runtime_tokenizer import RuntimeTokenizer


MIN_BARS = 64


@dataclass
class PredictionRecord:
    symbol: str
    timeframe: str
    timestamp: int
    artifact_name: str
    tokenizer_version: str
    next_coarse_id: int
    next_fine_id: int
    bounce_prob: float
    break_prob: float
    continuation_prob: float
    suggested_buffer_pct: float
    n_input_records: int
    n_bars_in_cache: int
    # Decoded geometry of the predicted next line.
    decoded_role: str = "unknown"
    decoded_direction: str = "flat"
    decoded_log_slope_per_bar: float = 0.0
    decoded_duration_bars: int = 1
    # Projected geometry of the predicted next LINE (not the trade).
    #   line_endpoint_pct_change = exp(decoded_slope * duration) - 1
    #     i.e. signed % change of the line's price endpoint vs anchor
    #     after `decoded_duration_bars` bars at the predicted slope.
    #     This is the LINE's path, NOT the trade's expected return —
    #     a breakout_long can have a negative line_endpoint_pct_change
    #     (descending resistance) and still be a long-direction trade.
    #   horizon_seconds = duration_bars * bar_seconds(timeframe)
    line_endpoint_pct_change: float = 0.0
    horizon_seconds: int = 0
    extras: dict = field(default_factory=dict)


class InferenceService:
    def __init__(
        self,
        manifest_path: Path | str,
        *,
        feature_cache: Optional[FeatureCache] = None,
        device: str | None = None,
    ):
        self.manifest = ArtifactManifest.load(manifest_path)
        cfg_dict = json.loads(Path(self.manifest.config_path).read_text(encoding="utf-8"))
        self.cfg = FusionConfig(**cfg_dict)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrendlineFusionModel(self.cfg).to(self.device).eval()
        state = torch.load(self.manifest.ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["state_dict"])

        self.tokenizer = RuntimeTokenizer(
            use_rule=self.cfg.use_rule_tokens,
            use_learned=self.cfg.use_learned_tokens,
            use_raw=self.cfg.use_raw_features,
            vqvae_path=self.cfg.vqvae_checkpoint_path,
        )
        # Hard fail if the runtime tokenizer doesn't match what produced
        # the training data: silent mismatch = wrong predictions.
        self.tokenizer.assert_version_matches(self.manifest.tokenizer_version)
        self.fc = feature_cache or FeatureCache(capacity=max(512, self.cfg.price_seq_len * 2))

    def push_bar(self, symbol: str, tf: str, bar: dict) -> bool:
        return self.fc.push(symbol, tf, bar)

    def predict(self, symbol: str, tf: str) -> Optional[PredictionRecord]:
        df = self.fc.bars_df(symbol, tf)
        if len(df) < MIN_BARS:
            return None

        # 1. Detect candidate lines on the in-cache history.
        lines = detect_lines(df, symbol=symbol, timeframe=tf,
                             max_lines=self.cfg.token_seq_len)
        # Right-align lines into the token slots; older lines pad on the left.
        lines = sorted(lines, key=lambda r: r.end_bar_index)[-self.cfg.token_seq_len:]

        # 2. Tokenise.
        tok = self.tokenizer.encode_records(lines)
        T = self.cfg.token_seq_len
        rule_c = np.zeros(T, dtype=np.int64); rule_c[T - len(lines):] = tok["rule_coarse"]
        rule_f = np.zeros(T, dtype=np.int64); rule_f[T - len(lines):] = tok["rule_fine"]
        l_c = np.zeros(T, dtype=np.int64); l_c[T - len(lines):] = tok["learned_coarse"]
        l_f = np.zeros(T, dtype=np.int64); l_f[T - len(lines):] = tok["learned_fine"]
        raw = np.zeros((T, self.cfg.raw_feat_dim), dtype=np.float32)
        if len(lines) > 0:
            raw[T - len(lines):] = tok["raw_feat"]
        token_pad = np.ones(T, dtype=bool); token_pad[T - len(lines):] = False

        # 3. Build the price window via the training-parity helper.
        price_window, price_pad = self.fc.price_window(symbol, tf, self.cfg.price_seq_len)

        # 4. Run the model.
        batch = {
            "price": torch.from_numpy(price_window).unsqueeze(0).to(self.device),
            "price_pad": torch.from_numpy(price_pad).unsqueeze(0).to(self.device),
            "rule_coarse": torch.from_numpy(rule_c).unsqueeze(0).to(self.device),
            "rule_fine": torch.from_numpy(rule_f).unsqueeze(0).to(self.device),
            "learned_coarse": torch.from_numpy(l_c).unsqueeze(0).to(self.device),
            "learned_fine": torch.from_numpy(l_f).unsqueeze(0).to(self.device),
            "raw_feat": torch.from_numpy(raw).unsqueeze(0).to(self.device),
            "token_pad": torch.from_numpy(token_pad).unsqueeze(0).to(self.device),
        }
        pred = self.model.predict(batch)

        # Decode the predicted next-line for chart overlay.
        from ..tokenizer.rule import (
            _decompose, SLOPE_COARSE_EDGES, SLOPE_FINE_QUANTILES, DURATION_EDGES,
        )
        from ..tokenizer.vocab import (
            LINE_ROLES, DIRECTIONS, coarse_cardinalities, fine_cardinalities,
        )
        coarse_id = int(pred["next_coarse_id"].item())
        fine_id = int(pred["next_fine_id"].item())
        ci = _decompose(coarse_id, coarse_cardinalities())
        fi = _decompose(fine_id, fine_cardinalities())
        decoded_role = LINE_ROLES[ci[0]] if 0 <= ci[0] < len(LINE_ROLES) else "unknown"
        decoded_dir = DIRECTIONS[ci[1]] if 0 <= ci[1] < len(DIRECTIONS) else "flat"
        dur_idx = ci[3]
        slope_coarse_idx = ci[4]
        slope_fine_idx = fi[0]
        # duration midpoint (bars)
        if dur_idx + 1 < len(DURATION_EDGES):
            dlo, dhi = DURATION_EDGES[dur_idx], DURATION_EDGES[dur_idx + 1]
            dmid = (dlo + dhi) / 2 if dhi < 1e6 else (dlo * 1.5 if dlo > 0 else 200)
        else:
            dmid = 50
        duration_bars = max(1, int(dmid))
        # slope midpoint (log per bar) within coarse x fine bin
        if slope_coarse_idx + 1 < len(SLOPE_COARSE_EDGES):
            s_lo = SLOPE_COARSE_EDGES[slope_coarse_idx]
            s_hi = SLOPE_COARSE_EDGES[slope_coarse_idx + 1]
            if s_hi > 1e6:
                s_hi = s_lo * 2 if s_lo > 0 else 0.08
            q_lo = s_lo + (s_hi - s_lo) * slope_fine_idx / SLOPE_FINE_QUANTILES
            q_hi = s_lo + (s_hi - s_lo) * (slope_fine_idx + 1) / SLOPE_FINE_QUANTILES
            abs_slope = (q_lo + q_hi) / 2
        else:
            abs_slope = 0.0
        signed_slope = abs_slope if decoded_dir == "up" else (
            -abs_slope if decoded_dir == "down" else 0.0
        )

        last_close = float(df["close"].iloc[-1]) if len(df) else 0.0
        last_open_time = int(df["open_time"].iloc[-1]) if len(df) else int(time.time() * 1000)

        # Real values for line_endpoint_pct_change + horizon_seconds.
        # NB: line_endpoint_pct_change is the LINE's path, not the trade's
        #     expected return. A breakout_long on a descending-resistance
        #     line has a NEGATIVE line_endpoint_pct_change.
        import math as _math
        try:
            line_endpoint_pct_change = float(_math.exp(signed_slope * duration_bars) - 1.0)
        except (OverflowError, ValueError):
            line_endpoint_pct_change = 0.0
        # bar seconds from timeframe string (e.g. "5m" -> 300)
        _tf_str = str(tf or "1m").lower()
        _tf_unit_secs = {"m": 60, "h": 3600, "d": 86400, "w": 86400 * 7}
        try:
            _n = int(_tf_str[:-1])
            _u = _tf_str[-1]
            bar_seconds = _n * _tf_unit_secs.get(_u, 60)
        except (ValueError, KeyError):
            bar_seconds = 60
        horizon_seconds = int(duration_bars * bar_seconds)

        return PredictionRecord(
            symbol=symbol.upper(), timeframe=tf,
            timestamp=last_open_time,
            artifact_name=self.manifest.artifact_name,
            tokenizer_version=self.manifest.tokenizer_version,
            next_coarse_id=int(pred["next_coarse_id"].item()),
            next_fine_id=int(pred["next_fine_id"].item()),
            bounce_prob=float(pred["bounce_prob"].item()),
            break_prob=float(pred["break_prob"].item()),
            continuation_prob=float(pred["continuation_prob"].item()),
            suggested_buffer_pct=float(pred["suggested_buffer_pct"].item()),
            n_input_records=len(lines),
            n_bars_in_cache=len(df),
            decoded_role=decoded_role,
            decoded_direction=decoded_dir,
            decoded_log_slope_per_bar=float(signed_slope),
            decoded_duration_bars=duration_bars,
            line_endpoint_pct_change=line_endpoint_pct_change,
            horizon_seconds=horizon_seconds,
            extras={"anchor_close": last_close, "anchor_open_time_ms": last_open_time},
        )
