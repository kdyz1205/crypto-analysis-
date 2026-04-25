"""Strategy catalog — predefined strategy templates users can browse and launch.

Each template defines: name, description, default config, supported timeframes.
No database needed — templates are code-defined, instances are stored in runtime JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class StrategyTemplate:
    template_id: str
    name: str
    name_en: str
    description: str
    category: str  # "trend" | "reversal" | "breakout" | "scalp"
    supported_timeframes: tuple[str, ...]
    default_trigger_modes: tuple[str, ...]
    default_params: dict[str, Any] = field(default_factory=dict)
    risk_level: str = "medium"  # "low" | "medium" | "high"


STRATEGY_CATALOG: list[StrategyTemplate] = [
    StrategyTemplate(
        template_id="sr_reversal",
        name="S/R 反转策略",
        name_en="S/R Reversal",
        description="在支撑/阻力区域寻找反转信号。价格触碰关键区域 + 出现拒绝 K 线形态时入场。止损在区域外侧，止盈在下一个反向区域。适合震荡市。",
        category="reversal",
        supported_timeframes=("5m", "15m", "1h", "4h", "1d"),
        default_trigger_modes=("rejection", "failed_breakout"),
        default_params={
            "lookback_bars": 80,
            "min_touches": 3,
            "rr_target": 2.0,
            "score_threshold": 0.62,
            "risk_per_trade": 0.003,
        },
        risk_level="medium",
    ),
    StrategyTemplate(
        template_id="sr_pre_limit",
        name="S/R 限价预挂策略",
        name_en="S/R Pre-Limit",
        description="在确认的支撑/阻力线附近预先挂限价单。价格接近线时自动入场。适合有耐心等待的交易者。胜率较低但盈亏比极高（5-10R）。",
        category="reversal",
        supported_timeframes=("15m", "1h", "4h"),
        default_trigger_modes=("pre_limit",),
        default_params={
            "lookback_bars": 80,
            "min_touches": 3,
            "rr_target": 2.0,
            "risk_per_trade": 0.002,
        },
        risk_level="low",
    ),
    StrategyTemplate(
        template_id="sr_retest",
        name="突破回测策略",
        name_en="Breakout Retest",
        description="等待价格突破支撑/阻力后，回调测试被突破的位置。旧支撑变新阻力（做空），旧阻力变新支撑（做多）。需要 wick rejection 确认。",
        category="breakout",
        supported_timeframes=("15m", "1h", "4h"),
        default_trigger_modes=("retest",),
        default_params={
            "lookback_bars": 80,
            "min_touches": 3,
            "rr_target": 2.5,
            "risk_per_trade": 0.003,
        },
        risk_level="medium",
    ),
    StrategyTemplate(
        template_id="sr_full",
        name="S/R 全模式策略",
        name_en="S/R Full Mode",
        description="同时启用所有触发模式：限价预挂 + 反转拒绝 + 假突破回收 + 突破回测。信号最多，覆盖面最广。适合模拟测试。",
        category="reversal",
        supported_timeframes=("5m", "15m", "1h", "4h"),
        default_trigger_modes=("pre_limit", "rejection", "failed_breakout", "retest"),
        default_params={
            "lookback_bars": 80,
            "min_touches": 3,
            "rr_target": 2.0,
            "risk_per_trade": 0.003,
        },
        risk_level="medium",
    ),
    StrategyTemplate(
        template_id="ma_ribbon",
        name="均线开花趋势策略",
        name_en="MA Ribbon Trend",
        description="多条均线（MA5/MA8/EMA21/MA55）按顺序排列形成'开花'形态时入场。上升排列做多，下降排列做空。适合趋势行情，ADX > 20 过滤。",
        category="trend",
        supported_timeframes=("15m", "1h", "4h", "1d"),
        default_trigger_modes=("pre_limit",),
        default_params={
            "lookback_bars": 100,
            "rr_target": 1.5,
            "risk_per_trade": 0.005,
            "atr_multiplier": 2.0,
            "adx_threshold": 20,
        },
        risk_level="medium",
    ),
    StrategyTemplate(
        template_id="hft_imbalance",
        name="盘口失衡反转",
        name_en="Queue Imbalance MR",
        description="检测盘口前5档买卖失衡。当一侧挂单量远超另一侧但价格推进衰减时，做反向1-3 tick的极短反转。使用 WebSocket 实时数据流。",
        category="scalp",
        supported_timeframes=("1m", "3m", "5m"),
        default_trigger_modes=("pre_limit",),
        default_params={"risk_per_trade": 0.005, "imbalance_threshold": 0.55, "target_ticks": 2},
        risk_level="high",
    ),
    StrategyTemplate(
        template_id="hft_sweep",
        name="微型突破跟随",
        name_en="Sweep Breakout",
        description="价格压缩后盘口被扫穿时跟随burst方向。检测深度真空+主动成交冲击+短时压缩区间突破。2-5 tick止盈。",
        category="scalp",
        supported_timeframes=("1m", "3m", "5m"),
        default_trigger_modes=("pre_limit",),
        default_params={"risk_per_trade": 0.005, "momentum_min": 2.0, "target_ticks": 3},
        risk_level="high",
    ),
    StrategyTemplate(
        template_id="hft_mm",
        name="做市偏斜策略",
        name_en="Inventory Market Making",
        description="同时在买卖两侧挂单赚价差。根据库存偏斜动态调整报价，根据波动/毒性调整点差宽度。Avellaneda-Stoikov模型。需要 WebSocket。",
        category="scalp",
        supported_timeframes=("1m", "3m", "5m"),
        default_trigger_modes=("pre_limit",),
        default_params={"risk_per_trade": 0.01, "min_spread_bps": 1.0, "max_inventory_usdt": 15.0},
        risk_level="high",
    ),
    StrategyTemplate(
        template_id="hf_scalp",
        name="高频剥头皮策略",
        name_en="HF Scalper",
        description="EMA(2/5) 交叉 + RSI(4) 过滤。1 分钟级别快进快出。止损 0.3%，止盈 0.5%。适合高波动币种。风险较高。",
        category="scalp",
        supported_timeframes=("1m", "3m", "5m"),
        default_trigger_modes=("pre_limit",),
        default_params={
            "rr_target": 1.67,
            "risk_per_trade": 0.01,
            "stop_pct": 0.003,
            "take_profit_pct": 0.005,
        },
        risk_level="high",
    ),
    StrategyTemplate(
        template_id="ma_ribbon_ema21_auto",
        name="MA Ribbon EMA21 自动",
        name_en="MA Ribbon EMA21 Auto",
        description=(
            "多 TF MA-ribbon 自动扫单。5m/15m/1h/4h 形成多头/空头排列时分层加仓 "
            "(Strategy Y 时间渐进)。SL 用当前 EMA21 × (1 ± buffer%) 跟随。"
            "策略级 -15% DD 自动 halt。详见 spec 2026-04-25-ma-ribbon-auto-execution."
        ),
        category="trend",
        supported_timeframes=("5m", "15m", "1h", "4h"),
        default_trigger_modes=("ribbon_formation",),
        default_params={
            "ribbon_buffer_pct": {"5m": 0.01, "15m": 0.04, "1h": 0.07, "4h": 0.10},
            "layer_risk_pct":    {"LV1": 0.001, "LV2": 0.0025, "LV3": 0.005, "LV4": 0.02},
            "max_concurrent_orders": 25,
            "per_symbol_risk_cap_pct": 0.02,
            "dd_halt_pct": 0.15,
            "directions": ["long", "short"],
        },
        risk_level="high",
    ),
]

CATALOG_BY_ID = {t.template_id: t for t in STRATEGY_CATALOG}


def list_templates() -> list[dict]:
    return [_template_to_dict(t) for t in STRATEGY_CATALOG]


def get_template(template_id: str) -> StrategyTemplate | None:
    return CATALOG_BY_ID.get(template_id)


def _template_to_dict(t: StrategyTemplate) -> dict:
    return {
        "template_id": t.template_id,
        "name": t.name,
        "name_en": t.name_en,
        "description": t.description,
        "category": t.category,
        "supported_timeframes": list(t.supported_timeframes),
        "default_trigger_modes": list(t.default_trigger_modes),
        "default_params": dict(t.default_params),
        "risk_level": t.risk_level,
    }


__all__ = [
    "STRATEGY_CATALOG",
    "StrategyTemplate",
    "get_template",
    "list_templates",
]
