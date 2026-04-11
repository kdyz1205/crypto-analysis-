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
