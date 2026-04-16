# 项目原则 (Principles)

> 这个文件是 AI 写代码必须遵守的不变法则。
> 任何违反这里规则的代码 = bug,无论它"看起来多合理"。
> AI 在每个任务开始前必须 cat 这个文件。

---

## 数据 / Data

**P1. 数据深度 = 交易所物理上限**
- 永远不要在前端 / 后端按 TF 给 days 加 cap (例如 `tfDays['15m']=90`)。
- HYPE 上市 482 天 → 1m/5m/15m/1h/4h/1d 全部都应该有 482 天 (Bitget 给多少给多少)。
- BTC 5 年都有 → 全部 TF 都拉满 5 年。
- 任何「为了加载快加 cap」的决定必须明示用户并征求同意,不能默默写进默认值。

**P2. CSV 缓存不能掩盖请求**
- 如果用户请求 365 天但 CSV 只有 90 天 → 必须重下,不能返回 CSV 那 90 天。
- CSV 是 cache,不是 source of truth。

---

## 画线 / Drawing

**P3. 画线后必须立即视觉反馈**
- 用户松手 → 线 0 延迟出现 (optimistic insert)。
- POST 在后台跑,失败回滚。
- 服务端 POST 必须 < 100ms,enrichment 推到 GET 时做。

**P4. 线永久存在,直到用户显式删除**
- 不允许 `max_age_seconds` 这类自动过期。
- 不允许 `max_distance_atr` 这类自动撤销。
- 不允许 watcher 自动「清理」用户的线。

**P5. 删除一条线 = 撤掉它的所有挂单**
- DELETE 线必须 cascade 取消所有 status in (pending, triggered) 的 conditional。
- 同时撤 Bitget 上的真实订单。

---

## 挂单 / Order

**P6. 方向 = 用户的明确选择**
- 不要用斜率 / 位置 / 任何启发式覆盖用户的方向选择。
- 用户点 LONG 就挂 LONG,即使我觉得"应该 SHORT"。
- 唯一的硬性拒绝:线离 mark 太远 (volatility-aware 阈值)。

**P7. 止损不是退出,是反手**
- 默认行为:止损触发 → spawn 反向 conditional (auto-reverse)。
- 这是项目的核心策略,不能改成普通 stop-out。
- `OrderConfig.reverse_*` 字段是这个逻辑的实现。

**P8. 推荐参数从历史统计得出,不从"合理值"猜**
- tolerance/stop/rr 必须从 ATR + 触点强度 + BB 状态 + 多 TF 一致性算出来。
- 默认 0.1% / 0.3% / 2.0 这种"看起来合理"的值是占位,不是 default。

**P9. 撤单必须三方同步**
- Bitget 撤了 → 本地 cond 同步 cancelled → watcher 不再 replan。
- 任何一方失败必须暴露给用户,不能 silent fail。

---

## 错误处理 / Error Handling

**P10. 错误必须可见**
- 不允许 `try / except: pass`。
- 不允许 silently `return None` 让 UI 显示空白。
- HTTP 错误必须把 detail 解析到 modal,不能只显示 "网络错误 HTTP 400"。

**P11. 状态必须可观测**
- 任何状态变更 (cond status / line update / order placed) 必须打 log。
- 任何被吞掉的 exception 必须 print + traceback。

---

## 缓存 / Cache

**P12. 轮询的端点必须 noCache**
- `account` / `pending_orders` / `mark_price` / `conditionals` / `drawings` 这些每秒/每 5s/每 10s 拉的端点,必须 `noCache: true`。
- 默认 30s GET cache 会让用户看到 30 秒前的余额、30 秒前的挂单状态。

**P13. CSV cache 必须按需失效**
- 见 P2。

---

## AI 工作模式 / AI Work Mode

**P14. 修 bug = 抽出原则 + grep 所有违反点 + 一次修完**
- 不允许「修当前症状,等用户发现下一个」。
- 听到一个 bug 描述 → 第一反应是「这个原则是什么 + 整个仓库有几处违反」。

**P15. 完成 = 枚举测试通过**
- 不允许「我修好了 (单 case 试过)」。
- 必须列出所有相关组合的测试结果,贴在回复里。

**P16. 写新代码前先删旧假设**
- 不允许「在旧的错误代码上叠新代码」。
- 任何 cap / limit / default 必须能回答「这是物理限制,还是我的偏见?」答不出来就不写。

**P17. 用户的明确诉求 > AI 的"常识"**
- 当用户说「不要 cap」,我的 TradingView-style 默认就是错的。
- 不要用「业界惯例」反驳用户的明确诉求。

---

## 修改这个文件

只有用户能加 / 改原则。AI 可以建议,但必须用户确认才能修改这个文件。
