# 策略 Runner 操作速查手册

> 适用范围:Phase 3 Pack A 上线后的 策略 Runner 页面
> 读者:真钱交易用户,快速查阅用
> 最后更新:2026-04-19

---

## 紧急情况一句话

> **出事了按红色"紧急平仓"按钮,输入 `FLATTEN`,一键平掉所有仓位并撤销所有挂单。**

---

## 一、按钮速查表

| 按钮 | 颜色 | 是否需要确认 | 作用范围 |
|------|------|-------------|---------|
| 启动 / 应用配置 | 蓝 | 仅 live 模式下需输入 `LIVE` | 启动或热更新 Runner |
| 立即扫描 | 灰 | 否 | 强制跑一次扫描 |
| 停止 | 橙 | 原生 confirm() 弹窗 | 停扫描,**不平仓** |
| 紧急平仓 | **红** | 需输入 `FLATTEN` | 平仓 + 撤单 |
| 重置 halt | 深红 | 需输入 `RESET` | 清除当日 DD 停机标记 |

---

## 二、按钮详细说明

### 1. 启动 / 应用配置

根据当前状态,这个按钮有三种行为:

- **Runner 正在运行** → 热更新配置(无需停机,新配置立即生效)
- **Runner 空闲 + dry_run=ON** → 立即冷启动(演练模式,不会下真单)
- **Runner 空闲 + dry_run=OFF + mode=live** → 弹出 **"真钱启动确认"** 模态框,必须手动输入大写 `LIVE` 才能启动

**什么时候用**:调整了 top_n / timeframes / 杠杆等参数后,点它让改动生效。
**什么时候不用**:已经在运行且你没改配置,点它只会白跑一次无效热更新。

---

### 2. 立即扫描

强制立刻跑一轮扫描 tick,**绕过 scan_interval_s 的等待**。

**什么时候用**:刚发行情突变、想立即让策略看一眼当前市场。
**什么时候不用**:Runner 没启动时点它无效;以及你每 10 秒点一次不会让策略更快,只会增加 API 负载。

---

### 3. 停止

调用 `stop_runner()`,**只停扫描循环,不会平仓,也不会撤单**。已经打开的仓位继续按 TP/SL 自行了结。

会弹出浏览器原生 `confirm()` 对话框,点"确定"生效。

**什么时候用**:想暂停新开仓,但让存量仓位跑完的正常收工。
**什么时候不用**:账户已经在出事、需要立刻减风险 — 那请用 **紧急平仓**,不要用停止。

---

### 4. 紧急平仓(红色)

命中后端 `/api/live-execution/flatten-all`:
- 市价平掉所有当前持仓
- 撤销所有未成交 plan orders(TP/SL 单等)

模态框要求输入大写 `FLATTEN`,避免手滑。

**什么时候用**:行情暴走、API 异常、自己判断错了、准备离线去吃饭等一切"不想再持仓"的情形。
**什么时候不用**:只是想停止开新单 — 用停止就好,不要动存量仓位。

---

### 5. 重置 halt(仅在 halt 状态可见)

模态框要求输入 `RESET`。清除**当日**的 daily-DD 停机标记,Runner 即可再次接受新开仓。

**什么时候用**:你评估过触发 halt 的回撤是个别事件(比如插针),想恢复交易。
**什么时候不用**:DD 是真的连续亏出来的 — 那就让它自然到 UTC 午夜再恢复,别硬撑。

⚠️ **重要:重置 halt 不会自动重启 Runner。** 你还得手动点"启动"。

---

## 三、Status Pill 颜色图例

<style>
.pill { display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; font-family:sans-serif; }
.pill-idle    { background:#9ca3af; color:#fff; }
.pill-running { background:#16a34a; color:#fff; }
.pill-stopped { background:#6b7280; color:#fff; }
.pill-halted  { background:#7f1d1d; color:#fff; }
</style>

| 状态 | 渲染 | 含义 |
|------|------|------|
| idle    | <span class="pill pill-idle">idle</span>       | Runner 从未启动或重置后空闲 |
| running | <span class="pill pill-running">running</span> | 正在扫描,可能有仓位 |
| stopped | <span class="pill pill-stopped">stopped</span> | 被人为点停止了 |
| halted  | <span class="pill pill-halted">HALTED · -5% TODAY</span> | 当日亏损触达 DD 上限,禁止开新仓 |

---

## 四、Halt 倒计时

只在 `halted` 状态下显示,格式:

```
自动恢复: 3h 47m (UTC 午夜)
```

到 UTC 00:00 自动清除 halt,**但 Runner 不会自动重启**,要手动点启动。不想等就用"重置 halt"。

---

## 五、Config 表单字段速记

| 字段 | 作用 | 建议值 |
|------|------|--------|
| top_n | 扫描池大小,按成交量排名取前 N 个币 | 30-50 |
| timeframes | 扫描的时间周期列表 | `["1h", "4h"]`(见 trendline TF decision) |
| scan_interval_s | 每轮扫描间隔(秒) | 60-300 |
| notional_usd | 单笔名义仓位美元 | 按风险预算填 |
| leverage | 杠杆倍数(合约) | 2-5 保守,>10 激进 |
| max_concurrent_positions | 同时持仓上限 | 5(见 trendline 实盘 sizing) |
| dry_run | 是否演练模式(不下真单) | 新配置上线前先 ON 跑几天 |

---

## 六、FAQ

**Q1:我点了紧急平仓但当前没任何仓位,会报错吗?**
不会。后端 `flatten-all` 空仓位空挂单时返回 200 OK,前端弹个 "已平仓 0 个" 提示,完事。

**Q2:halt 被重置后 Runner 会自动重启扫描吗?**
**不会。** 重置只是清标记。Runner 状态仍然是 stopped / idle,你必须自己点"启动"才会恢复扫描。

**Q3:我点了停止,但价格继续跑,原来的仓位会怎样?**
仓位和 TP/SL 单都还在交易所那边挂着,按原定止盈止损自己了结。停止只关掉本地扫描循环。要想一起清掉,请改用紧急平仓。

**Q4:dry_run=ON 时点启动,会不会不小心下到真单?**
不会。dry_run 开着时冷启动不需要 `LIVE` 确认,所有信号只写日志、不发交易所请求。确认真钱下单的 `LIVE` 模态框只在 dry_run=OFF 且 mode=live 时才出现。

**Q5:我手滑在模态框输错了字母,会怎样?**
提交按钮保持禁用状态。必须**完全一致、大写、无空格**才激活。输错了直接关掉模态框重来就行,不会触发任何操作。

---

## 七、紧急情况一览(给慌乱时看的)

| 症状 | 该按什么 |
|------|---------|
| 行情暴走我想立刻跑路 | **紧急平仓**(FLATTEN) |
| 只是今天不想再开新仓,存量让它跑 | 停止 |
| 触达 DD 但我确定是插针想继续 | 重置 halt(RESET) + 启动 |
| 改了配置想立即生效 | 启动 / 应用配置 |
| 刚出新闻想立刻扫一次 | 立即扫描 |

---

**原则:红色按钮 = 真实动账户。出事优先红色,别犹豫。**
