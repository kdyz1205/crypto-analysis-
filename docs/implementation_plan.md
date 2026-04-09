# Trendline Bot 实施计划

更新时间：2026-04-08

## 目标与边界

本计划针对当前 `crypto-analysis-` 仓库，规划一条“不依赖 TradingView、以后端 Python 为唯一真相源”的趋势线自动交易实现路径：

`Bitget 数据 -> 市场数据层 -> 趋势线识别层 -> 信号状态机 -> 风控层 -> 执行层 -> 前端图表渲染 -> bot 自动下单`

补充说明：

1. 用户已明确当前实际接入的是 Bitget 行情。
2. 因此本计划改为 `Bitget-first`。
3. 策略核心仍然必须保持交易所无关，只消费标准化后的 OHLCV / symbol metadata / precision / order status。
4. 当前仓库的现实情况是：市场数据与 live execution 代码都偏 OKX，需要先补 Bitget 适配层，不能把现有 OKX 假设直接带入后续阶段。

阶段 1 只做审计与计划，不改现有业务逻辑。后续实施必须遵守：

1. 前端只渲染，不做策略计算。
2. 回测、逐 bar 重放、实时交易共用同一套核心策略函数。
3. 先做纯策略核心，再接回放，再接图表，再做 paper trading，最后才接 live。
4. 新模块优先并行挂接到现有系统，避免把当前 V6 agent 和新趋势线 bot 混写在一起。

## 当前仓库结构摘要

### 后端

- `server/app.py`
  - FastAPI 入口，负责 router 注册、单例初始化、事件订阅器启动。
- `server/data_service.py`
  - 当前主市场数据层。
  - 当前实现偏 OKX。
  - 已支持 OKX 历史/实时补尾、symbol 元数据、OHLCV 规范化、缓存、resample。
- `server/pattern_service.py`
  - 对 `sr_patterns.py` 的服务包装，给前端输出支撑阻力/趋势线/形态数据。
- `sr_patterns.py`
  - 大型模式识别引擎，含 pivot、趋势线、三角形、consolidation、replay cutoff 等能力。
- `server/agent_brain.py`
  - 当前 V6 自动交易主脑。
  - 已包含：信号生成、pre-trade checklist、仓位管理、审计日志、Telegram 通知、事件发布。
- `server/okx_trader.py`
  - 当前执行与账户状态核心。
  - 已包含 paper/live 双模式、风险限制、账户状态、OKX REST 下单/平仓。
- `server/core/events.py`
  - 进程内事件总线，已支持 `signal.* / order.* / position.* / risk.* / agent.* / ops.* / summary.*`。
- `server/core/state_machines.py`
  - 通用 signal lifecycle 状态机，适合后续复用到趋势线信号执行生命周期。
- `server/routers/*.py`
  - 已按 `market / patterns / research / agent / risk / execution / stream / ops` 分层。

### 前端

- `frontend/index.html` + `frontend/app.js`
  - v1 单文件前端，逻辑较重，包含绘图和老的图表交互。
- `frontend/v2.html` + `frontend/js/*`
  - v2 模块化前端。
  - `frontend/js/workbench/chart.js` 是当前最清晰的图表接入点。
  - `frontend/js/workbench/patterns.js` 负责把后端给的线段绘制到 Lightweight Charts。
  - `frontend/js/services/stream.js` 已接入 SSE。
  - `frontend/js/execution/panel.js` 已有执行中心 UI 壳子。

### 测试与文档

- `tests/`
  - 当前只有非常轻量的接口测试，主要覆盖 `/api/health` 与 `/api/symbols`。
- `docs/superpowers/plans/*.md`
  - 已有架构边界与前端分层文档，可作为实现约束参考。

## 已有模块可复用清单

### 可以直接复用

#### 1. 市场数据层

- `server/data_service.py`
  - 适合作为新策略唯一 candle 输入源的“标准化入口”。
  - 已具备：
    - OKX instrument/symbol 获取
    - OHLCV 下载与缓存
    - 多周期 resample
    - `/api/ohlcv` 与 `/api/chart` 所需数据序列化

结论：新趋势线策略不应该直接绑定 OKX，但可以复用这里的缓存、resample、返回格式和 API 壳层。
需要在此基础上补一个 Bitget market adapter，把 Bitget candles/symbol metadata 归一化到现有 schema，再交给策略核心。

#### 2. FastAPI 壳层与路由组织

- `server/app.py`
- `server/routers/*`
- `server/core/dependencies.py`

结论：后续只需新增 `server/routers/strategy.py` 和少量 schema/model 文件，无需重做应用壳层。

#### 3. 事件总线与推送通道

- `server/core/events.py`
- `server/routers/stream.py`
- `server/subscribers/sse_broadcast.py`
- `server/subscribers/audit.py`

结论：趋势线 bot 的运行事件、订单状态、前端标记更新，优先走现有 event bus + SSE，不要新开一套 websocket 基础设施。

#### 4. 图表渲染基础设施

- `frontend/js/workbench/chart.js`
- `frontend/js/workbench/patterns.js`
- `frontend/js/state/market.js`
- `frontend/js/services/market.js`
- `frontend/js/services/patterns.js`

结论：前端已有 Lightweight Charts 和 overlay 绘制能力，后续新增 overlay 模块即可，不需要改图表库。

#### 5. 现有回测工具函数

- `server/backtest_service.py`
  - `_sma/_ema/_atr` 等基础指标函数可直接复用。

结论：如果新策略需要 ATR 容差、止损距离、波动过滤，可以直接复用这些纯函数。

### 可以“选择性复用”，但不建议直接依赖为核心

#### 1. `sr_patterns.py`

可借鉴：

- pivot / trendline / tolerance / replay cutoff 的思路
- 部分 dataclass 结构
- “严格验证”和 replay 限制的经验

不建议直接作为新 bot 核心依赖，原因：

1. 文件体积很大，职责过多，既做检测又做可视化兼容。
2. 当前输出更偏前端展示，不是为“信号状态机 + 执行层”设计。
3. 新策略需要明确定义三触确认、armed 区、failed breakout、signal_id、invalidation state，这些在现有文件里不是一等公民。

结论：可以参考算法细节，但新的自动交易核心应单独建包。

#### 2. `server/agent_brain.py`

可复用：

- `PreTradeChecklist` 的工程化思路
- structured audit logging 的习惯
- 事件发布时间点

不建议直接把趋势线策略塞进去，原因：

1. 当前 `AgentBrain` 已将“策略 + 风控 + 执行 + 账户状态”强耦合。
2. 当前信号逻辑是 V6 MA/BB 体系，不适合继续堆叠第二套核心策略。
3. 新策略需要 replay 与实时共用函数，如果直接拼在这里，边界会继续变脏。

结论：趋势线 bot 应并行新增模块，后续再通过统一执行入口接入。

#### 3. `server/okx_trader.py`

可复用：

- OKX 签名、价格查询、余额检查、live 下单基础逻辑

当前问题：

1. 同时承担 paper/live execution、账户状态、风险限制。
2. `AgentState` 与 `RiskLimits` 都和单一 agent 生命周期耦合。

结论：如果后续 live execution 也走 Bitget，那么 `server/okx_trader.py` 只能作为“执行器参考实现”，不应继续作为目标执行通道。
阶段 6 先做独立 paper execution；阶段 7 再新增 Bitget execution adapter。

## 必须新增的模块

### 1. 纯策略核心包

建议新增 `server/strategy/`，内部只放纯函数和纯数据结构：

- `server/strategy/config.py`
- `server/strategy/types.py`
- `server/strategy/pivots.py`
- `server/strategy/trendlines.py`
- `server/strategy/scoring.py`
- `server/strategy/signals.py`
- `server/strategy/replay.py`
- `server/strategy/state_machine.py`

职责：

- 只接收 candles 和配置
- 只返回结构化结果
- 不直接访问交易所、数据库、路由、前端

### 1.5. Bitget 数据适配层

建议新增：

- `server/market/bitget_client.py`
- `server/market/bitget_adapter.py`

职责：

- 获取 Bitget 历史 candles / 最新 candles / symbol 元数据 / 价格精度
- 把 Bitget 返回归一化为当前后端统一 candle schema
- 对上游策略层隐藏交易所差异

### 2. 策略 API 层

- `server/routers/strategy.py`
- `server/schemas/strategy.py`

职责：

- 对外暴露 snapshot、replay、state、overlay 数据
- 为前端图层与后续 paper/live engine 提供统一入口

### 3. 执行与风控拆分层

建议新增：

- `server/execution/types.py`
- `server/execution/order_manager.py`
- `server/execution/position_manager.py`
- `server/execution/bitget_trader.py`
- `server/risk/risk_rules.py`
- `server/risk/kill_switch.py`

职责：

- 把“下单状态机 / 持仓状态 / client_order_id / 幂等 / 熔断”从当前 `okx_trader.py` 中解耦出来

### 4. 前端 overlay 模块

建议新增：

- `frontend/js/workbench/overlays/trendline_overlay.js`
- `frontend/js/workbench/overlays/signal_overlay.js`
- `frontend/js/workbench/overlays/order_overlay.js`
- `frontend/js/state/strategy.js`
- `frontend/js/services/strategy.js`

职责：

- 只消费后端结构化结果
- 控制图层开关
- 不做任何策略判定

## 建议的新目录结构

```text
server/
  market/
    __init__.py
    bitget_client.py
    bitget_adapter.py
  strategy/
    __init__.py
    config.py
    types.py
    pivots.py
    trendlines.py
    scoring.py
    signals.py
    replay.py
    state_machine.py
  execution/
    __init__.py
    types.py
    order_manager.py
    position_manager.py
    bitget_trader.py
  risk/
    __init__.py
    risk_rules.py
    kill_switch.py
  schemas/
    __init__.py
    strategy.py
  routers/
    strategy.py

frontend/
  js/
    services/
      strategy.js
    state/
      strategy.js
    workbench/
      overlays/
        trendline_overlay.js
        signal_overlay.js
        order_overlay.js

tests/
  strategy/
    test_pivots.py
    test_trendlines.py
    test_signals.py
    test_replay.py
    test_state_machine.py
  execution/
    test_order_manager.py
    test_position_manager.py
    test_risk_rules.py
```

说明：

- 新逻辑优先放到 `server/strategy`、`server/execution`、`server/risk`，不要继续堆到 `agent_brain.py`。
- 阶段 1 不需要迁移旧文件，只做并行新增。
- 阶段 6 之前，不建议大规模重命名 `okx_trader.py`，避免引入无关重构。
- 更稳的做法是并行新增 Bitget adapter，而不是先去改老的 OKX 文件名和调用链。

## 需要新增的数据结构定义

以下结构建议作为后端唯一标准，供 replay、API、前端、paper/live 执行共享。

### 1. Candle 引用

优先复用当前 `data_service.py` 返回的 OHLCV schema；策略内部可标准化为：

```python
class Candle(TypedDict):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
```

### 2. Pivot

```python
@dataclass
class Pivot:
    pivot_id: str
    kind: Literal["high", "low"]
    index: int
    time: int
    price: float
    left_bars: int
    right_bars: int
    confirmed_at_index: int
```

关键点：

- 必须显式记录 `confirmed_at_index`，避免未来函数污染。

### 3. TrendlineCandidate / ActiveTrendline

```python
@dataclass
class Trendline:
    line_id: str
    side: Literal["resistance", "support"]
    state: Literal["candidate", "confirmed", "armed", "triggered", "invalidated", "closed"]
    anchor_pivot_ids: list[str]
    touch_pivot_ids: list[str]
    x1: int
    y1: float
    x2: int
    y2: float
    slope: float
    intercept: float
    touch_count: int
    score: float
    tolerance_abs: float
    confirmed_at_index: int | None
    invalidated_at_index: int | None
```

### 4. ProjectedLinePrice

```python
@dataclass
class ProjectedLinePrice:
    line_id: str
    bar_index: int
    time: int
    price: float
```

### 5. StrategySignal

```python
@dataclass
class StrategySignal:
    signal_id: str
    line_id: str
    symbol: str
    interval: str
    side: Literal["long", "short"]
    trigger_mode: Literal["pre_limit", "rejection", "failed_breakout"]
    state: Literal["candidate", "confirmed", "armed", "triggered", "invalidated", "closed", "blocked"]
    bar_index: int
    time: int
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason_code: str
```

### 6. ReplaySnapshot

```python
@dataclass
class ReplaySnapshot:
    symbol: str
    interval: str
    bar_index: int
    time: int
    pivots: list[Pivot]
    active_trendlines: list[Trendline]
    projected_prices: list[ProjectedLinePrice]
    signals: list[StrategySignal]
    invalidations: list[dict]
    positions: list[dict]
    orders: list[dict]
```

### 7. OrderIntent / OrderState / PositionState

```python
@dataclass
class OrderIntent:
    order_intent_id: str
    signal_id: str
    client_order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    order_type: Literal["limit", "market"]
    price: float | None
    quantity: float
    stop_loss: float | None
    take_profit: float | None
```

```python
@dataclass
class PositionState:
    position_id: str
    signal_id: str
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    state: Literal["open", "closing", "closed"]
```

## 前端 chart 接入点

### 推荐主接入点

- `frontend/js/workbench/chart.js`

原因：

1. 已是 v2 图表主入口。
2. 已负责 candle 数据加载、基础 overlay 绘制、symbol/timeframe 切换。
3. 当前逻辑已经把图层渲染与数据请求分开，适合继续接入新的 strategy overlay service。

### 相关配套接入点

- `frontend/js/services/market.js`
  - 继续负责 candle/market 数据。
- `frontend/js/services/stream.js`
  - 继续负责 SSE 事件流。
- `frontend/js/state/market.js`
  - 可继续存当前 symbol/interval/candle。
- `frontend/js/workbench/patterns.js`
  - 可借鉴绘制方式，但趋势线 bot 的 overlay 最好拆到单独模块，不要继续混入 S/R 图层。

### 不推荐优先接入点

- `frontend/app.js`

原因：

1. v1 是单文件旧实现，逻辑密度高。
2. 如果在这里扩展趋势线 bot，可维护性会明显变差。

结论：趋势线 bot 优先接入 `/v2` 页面，v1 只做兼容，不做主战场。

## 现有 websocket / api / candle 流复用方案

### Candle 流

复用：

- `server/data_service.py`
- `/api/ohlcv`
- `/api/chart`

建议：

- 趋势线策略内部直接使用“标准化后的” `get_ohlcv_with_df(...)` 作为数据入口。
- replay 也用同一批 candle 数据，不允许另开一套 CSV 读取逻辑。

补充：

- 在用户当前的 Bitget 路线下，`get_ohlcv_with_df(...)` 需要先具备 Bitget 数据源能力，或者由上层包一层 Bitget adapter 后再复用。

### API 层

新增而不是改写原接口：

- `GET /api/strategy/snapshot`
  - 返回当前 bar 的趋势线状态、touch 点、armed 区、signals、order markers。
- `GET /api/strategy/replay`
  - 返回逐 bar 快照，可分页或限制 bars。
- `GET /api/strategy/config`
  - 返回当前策略参数。
- `POST /api/strategy/config`
  - 更新策略参数。

这样做的好处：

1. 不破坏现有 `/api/patterns` 和 `/api/chart` 的用户路径。
2. 新旧策略可以并行存在。

### 实时事件流

复用现有 SSE：

- `/api/stream`
- `server/core/events.py`

建议新增事件类型：

- `strategy.line.confirmed`
- `strategy.line.invalidated`
- `strategy.signal.armed`
- `strategy.signal.triggered`
- `strategy.signal.cancelled`
- `order.intent.created`
- `order.timeout.cancelled`
- `position.updated`

注意：

- 现有 `audit.py`、`sse_broadcast.py`、`stream.js` 目前只订阅既定前缀。
- 如果新增 `strategy.*` 前缀，需要同步更新 subscriber 注册与前端监听列表。

## 分阶段实施顺序

以下顺序以“最小侵入、便于 review”为原则，和当前仓库最匹配。

### 阶段 2：冻结策略规格

输入：

- 本实施计划
- 当前仓库数据格式与事件流约束

输出：

- `docs/strategy_spec.md`

验收标准：

- 明确定义 pivot 确认规则
- 明确定义触碰、确认、失效、trigger mode
- 明确参数列表与默认值
- 明确实时与回测共享同一核心函数

主要风险：

- 规则定义不严导致后续实现阶段频繁返工
- 未明确 future leak 约束

### 阶段 2.5：Bitget 数据接入与标准化

输入：

- 当前 `server/data_service.py`
- Bitget 行情接口返回格式

输出：

- Bitget market adapter
- 统一 candles / symbol info / precision schema

验收标准：

- 能从 Bitget 拉取历史和近实时 candles
- 输出格式与后续策略核心完全解耦，不把 Bitget 特殊字段泄漏进策略层
- `/api/ohlcv` 或等效内部入口可用同一 schema 返回数据

主要风险：

- 直接把 Bitget 字段结构渗透进策略函数
- 数据源切换时前端或 replay 使用了不同 schema

### 阶段 3：实现纯策略核心

输入：

- `docs/strategy_spec.md`
- 标准化后的 candles schema

输出：

- `server/strategy/*`
- 纯函数测试用例

验收标准：

- 给定同一组 candles，能稳定输出 pivots / trendlines / scores / projected prices / signals / invalidation states
- 不依赖 Bitget、路由、数据库、前端
- 单元测试可跑

主要风险：

- 直接调用 `sr_patterns.py` 导致边界模糊
- pivot 确认仍然偷偷看未来

### 阶段 4：实现逐 bar 历史回放器

输入：

- 纯策略核心
- candles 历史序列

输出：

- `server/strategy/replay.py`
- `server/strategy/state_machine.py`
- replay 测试

验收标准：

- 每个 bar 都能产出状态快照
- 回放结果与实时逻辑共用同一核心函数
- 可导出 `jsonl` 或 `csv`

主要风险：

- 为了提速而把未来 bars 带入当前状态
- replay 状态机与实时状态机双写

### 阶段 5：接入前端图表可视化

输入：

- replay/snapshot 输出结构
- v2 chart 模块

输出：

- `server/routers/strategy.py`
- `server/schemas/strategy.py`
- `frontend/js/services/strategy.js`
- `frontend/js/state/strategy.js`
- `frontend/js/workbench/overlays/*`

验收标准：

- 前端不做策略计算
- 可显示：
  - active trendlines
  - touch points
  - projected line price
  - armed zones
  - signal markers
  - invalidation markers
  - order markers
  - stop/tp markers
- 图层可开关

主要风险：

- 把策略判断塞回前端
- 直接复用旧 `patterns.js` 导致 S/R 与 bot overlay 混杂

### 阶段 6：实现 paper trading 执行层

输入：

- 策略信号事件
- 风控规则

输出：

- `server/execution/*`
- `server/risk/*`
- 订单/持仓状态机测试

验收标准：

- 支持 signal -> risk -> intent -> order -> position -> close 全链路
- 支持幂等、重复订单拦截、最大仓位、单笔风险、日亏熔断
- 默认只跑 paper

主要风险：

- 继续沿用 `OKXTrader` 的强耦合模式，导致 paper/live 难拆
- 在 paper 阶段就提前耦合 Bitget live API
- signal_id 与 client_order_id 体系不稳定

### 阶段 7：接入 live trading，默认关闭

输入：

- 稳定的 paper trading 执行层
- 已验证的 risk engine

输出：

- live adapter
- 对账与恢复逻辑
- 环境变量门控

验收标准：

- 默认关闭
- 必须同时满足：
  - `ENABLE_LIVE_TRADING=true`
  - `DRY_RUN=false`
  - `CONFIRM_LIVE_TRADING=true`
- 所有 live order 都经过 risk engine

主要风险：

- 误发单
- 启动恢复不完整导致本地状态与交易所状态漂移

## 逐阶段输入 / 输出 / 验收 / 风险总表

| 阶段 | 输入 | 输出 | 验收标准 | 风险点 |
|---|---|---|---|---|
| 1 审计 | 现有仓库 | `docs/implementation_plan.md` | 只新增文档 | 误判现有边界 |
| 2 规格 | 实施计划 | `docs/strategy_spec.md` | 规则可直接编码 | 未来函数污染未定义 |
| 2.5 Bitget 数据 | Bitget API + 当前数据层 | Bitget adapter | 返回统一 schema | 数据源字段泄漏 |
| 3 核心 | strategy spec + candles | `server/strategy/*` | 纯函数+测试 | 复用旧模式识别过深 |
| 4 replay | 纯核心 | `replay/state_machine` | 逐 bar 无偷看未来 | 实时/回测双写 |
| 5 图表 | replay/snapshot API | v2 overlays | 前端只渲染 | 前后端职责反转 |
| 6 paper | signal + risk + execution types | paper engine | 全链路可模拟 | 订单幂等缺失 |
| 7 live | 稳定 paper + Bitget adapter | gated live engine | 默认关闭且可验证 | 误发真实订单 |

## 对当前仓库的关键判断

### 1. 最适合的“主战场”是 v2，不是 v1

原因：

- v2 已模块化。
- 图表、SSE、执行中心、研究面板已经拆分。
- 更适合趋势线 bot 的结构化接入。

### 2. 新趋势线 bot 不应直接写进 `server/agent_brain.py`

原因：

- 当前 `agent_brain.py` 已经是单策略 V6 中心。
- 再叠加趋势线策略会让策略层、执行层、账户层进一步耦合。

建议：

- 先把趋势线核心做成独立包。
- 阶段 6 再决定是“新建 trendline engine”还是“把旧 agent 抽象成可插拔 strategy runner”。

### 3. 现有 SSE 和 event bus 值得继续用

原因：

- 已经有从后端到前端的事件通路。
- 非常适合推送 signal/order/position/invalidation。

### 4. `sr_patterns.py` 适合参考，不适合作为最终自动下单内核

原因：

- 更像研究/识别/可视化引擎，不是稳定、窄职责的执行策略核心。

## 推荐实施策略

### 最小侵入路线

1. 保持现有 V6 agent 正常工作。
2. 先补 `Bitget -> 标准化 candles schema` 的适配层。
3. 单独新增 `server/strategy` 纯核心。
4. 单独新增 `server/routers/strategy.py` 给 v2 图表用。
5. 单独新增 `server/execution` 与 `server/risk`，先跑 paper。
6. paper 跑稳后，再接 Bitget live adapter。

### 暂不建议做的事

1. 阶段 3 之前重构 `okx_trader.py`
2. 阶段 3 之前改造 `agent_brain.py` 为多策略引擎
3. 阶段 5 之前在前端加任何策略计算
4. 阶段 6 之前直接打开 live trading

## 本阶段结论

当前仓库已经具备以下基础：

- 可复用的市场数据壳层
- FastAPI 路由层
- 事件总线与 SSE 推送
- v2 图表与执行中心 UI 壳层
- 基础 paper/live execution 经验代码

缺失的是一条真正解耦的“趋势线策略核心 -> replay -> strategy API -> paper execution -> live adapter”主线。

在用户当前的 Bitget 路线下，还额外缺一个“Bitget -> 标准化市场数据/执行适配层”。

因此，后续最稳的实施方式不是继续往旧 V6 agent 里堆逻辑，而是新增一条清晰主线，并在 router、event bus、v2 chart 层与现有系统对接。
