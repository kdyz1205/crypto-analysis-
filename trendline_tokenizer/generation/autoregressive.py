"""Phase 3: Autoregressive next-N trendline generator.

Given a trained TrendlineFusionModel and an initial token-sequence state,
sample N future trendlines step-by-step. At each step we feed the model
the current sequence, take the predicted (coarse, fine) token, decode it
into a TrendlineRecord via the rule tokenizer, then shift the new token
into the sequence and repeat.

Why this matters (per the user's Phase 3 spec): the model already learns
a 1-step `next_coarse` head; multi-step generation lets it project a
*sequence* of structures (e.g. "support → break → retest → new resistance"),
which is what the discretionary trader actually wants to see.

Sampling modes:
  - greedy:        argmax (deterministic, baseline)
  - temperature:   softmax / T then sample
  - top_k:         restrict sampling to k highest logits
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from ..models.full_model import TrendlineFusionModel
from ..schemas.trendline import TokenizedTrendline, TrendlineRecord
from ..tokenizer.rule import decode_rule


@dataclass
class GeneratedStep:
    """One autoregressive step's output."""
    step: int
    rule_coarse_id: int
    rule_fine_id: int
    bounce_prob: float
    break_prob: float
    continuation_prob: float
    suggested_buffer_pct: float
    decoded_record: TrendlineRecord
    # For diagnostics: the top-5 next_coarse_id candidates the model
    # considered (to surface uncertainty in the UI / backtests).
    top5_coarse_ids: list[int] = field(default_factory=list)
    top5_coarse_probs: list[float] = field(default_factory=list)


class AutoregressiveGenerator:
    """Sample N future trendlines from a trained TrendlineFusionModel.

    The model itself only predicts *one* next token per forward pass; the
    generator wraps that into a streaming loop where each sampled token
    is fed back as the most-recent context for the next step.
    """

    def __init__(self, model: TrendlineFusionModel, *, device: str | torch.device | None = None):
        self.model = model
        self.cfg = model.cfg
        self.device = device or next(model.parameters()).device

    # ── sampling helpers ────────────────────────────────────────────

    @staticmethod
    def _sample_logits(
        logits: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample one index per row from `logits` (B, V)."""
        if temperature <= 0:
            return logits.argmax(dim=-1)
        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None and top_k > 0 and top_k < logits.size(-1):
            top_v, top_i = logits.topk(top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, top_i, top_v)
            logits = mask
        probs = logits.softmax(dim=-1)
        # multinomial samples one index per row
        return torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1)

    # ── single-step roll forward ───────────────────────────────────

    def _shift_in_token(
        self, batch: dict, coarse_id: torch.Tensor, fine_id: torch.Tensor
    ) -> dict:
        """Append (coarse_id, fine_id) to the token sequence, shifting
        left if at max length. We do NOT have actual learned/raw features
        for the newly-sampled (still hypothetical) trendline, so those
        slots are zeroed out — the model will rely on its rule-token
        embeddings + the residual price context for the next prediction.
        """
        new_batch = {k: v for k, v in batch.items()}
        T = self.cfg.token_seq_len
        for key in ("rule_coarse", "rule_fine",
                    "learned_coarse", "learned_fine", "raw_feat",
                    "token_pad"):
            if key not in new_batch:
                continue
            cur = new_batch[key]
            # cur shape: (B, T) for ids/pad, (B, T, F) for raw_feat
            if cur.dim() == 2:
                shifted = torch.cat([cur[:, 1:], cur[:, :1]], dim=1)  # placeholder
                if key == "rule_coarse":
                    shifted[:, -1] = coarse_id
                elif key == "rule_fine":
                    shifted[:, -1] = fine_id
                elif key in ("learned_coarse", "learned_fine"):
                    shifted[:, -1] = 0    # unknown; embedding[0] used
                elif key == "token_pad":
                    shifted[:, -1] = False  # the new token is real (not pad)
                new_batch[key] = shifted
            elif cur.dim() == 3:                      # raw_feat
                shifted = torch.cat(
                    [cur[:, 1:, :], torch.zeros_like(cur[:, :1, :])], dim=1
                )
                new_batch[key] = shifted
        return new_batch

    @torch.no_grad()
    def generate(
        self,
        batch: dict,
        *,
        n_steps: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
        reference_record: TrendlineRecord | None = None,
    ) -> list[GeneratedStep]:
        """Sample n_steps future trendlines autoregressively.

        batch: a forward-ready batch with shape (B=1, ...). Multi-batch
               generation works but the convenience decode_rule call
               assumes B=1; for B>1, generated_record on each step uses
               row 0 by convention.
        n_steps: how many steps to roll forward.
        temperature: <=0 → argmax (deterministic).
        top_k: optional truncation.
        seed: torch RNG seed for reproducibility.
        reference_record: optional record whose symbol/exchange/timeline
            anchors are carried into each decoded record.
        """
        if n_steps <= 0:
            return []

        self.model.eval()
        rng = None
        if seed is not None:
            rng = torch.Generator(device=self.device)
            rng.manual_seed(seed)

        # Move batch to device
        batch = {k: v.to(self.device) if hasattr(v, "to") else v
                 for k, v in batch.items()}

        out_steps: list[GeneratedStep] = []
        cur = batch
        for step in range(n_steps):
            logits = self.model(cur)
            coarse_logits = logits["next_coarse_logits"]   # (B, Vc)
            fine_logits = logits["next_fine_logits"]       # (B, Vf)

            coarse_id = self._sample_logits(
                coarse_logits, temperature=temperature, top_k=top_k, rng=rng,
            )
            fine_id = self._sample_logits(
                fine_logits, temperature=temperature, top_k=top_k, rng=rng,
            )

            # Per-step diagnostics
            probs = coarse_logits[0].softmax(dim=-1)
            top5_v, top5_i = probs.topk(min(5, probs.size(-1)))

            # Decode the FIRST row of the batch into a concrete record
            tok = TokenizedTrendline(
                record_id=f"gen-step-{step}",
                coarse_token_id=int(coarse_id[0].item()),
                fine_token_id=int(fine_id[0].item()),
                tokenizer_version="rule.v1",
            )
            decoded = decode_rule(tok, reference_record=reference_record)

            bp = float(logits["bounce_logits"][0].softmax(dim=-1)[1].item())
            brkp = float(logits["break_logits"][0].softmax(dim=-1)[1].item())
            cop = float(logits["continuation_logits"][0].softmax(dim=-1)[1].item())
            buf = float(logits["buffer_pct"][0].item())

            out_steps.append(GeneratedStep(
                step=step,
                rule_coarse_id=int(coarse_id[0].item()),
                rule_fine_id=int(fine_id[0].item()),
                bounce_prob=bp,
                break_prob=brkp,
                continuation_prob=cop,
                suggested_buffer_pct=buf,
                decoded_record=decoded,
                top5_coarse_ids=[int(i.item()) for i in top5_i],
                top5_coarse_probs=[float(v.item()) for v in top5_v],
            ))

            # Roll the sampled token into the context for next step
            cur = self._shift_in_token(cur, coarse_id, fine_id)

        return out_steps
