# Review Harness

每个 phase 完成后，必须自动执行：

`Implementer -> Reviewer -> 修复（如需要）-> 再 Reviewer`

除非用户明确要求跳过，否则不得省略。

## 总原则

1. 先实现，再审查。
2. Reviewer 必须把自己当成“另一个人”，默认实现里有 bug、歧义、状态断裂或架构污染。
3. 测试通过不是自动 PASS，只是审查输入之一。
4. 只要存在 phase 级别 blocking issue，必须先修复，再复审。
5. 输出必须工程化，只写结论、证据、风险、修复，不写冗长思维过程。

## 固定流程

### Step 1. 读取本 phase 目标

- 明确允许修改哪些文件
- 明确禁止修改哪些模块
- 明确验收标准

### Step 2. Implementer 实现

- 只做本 phase 必需内容
- 不扩大范围
- 写必要测试
- 跑测试 / 语法检查 / 编译检查

### Step 3. Implementer 初稿 review 包

必须包含：

- 改动文件
- 关键实现点
- 运行方式
- 测试结果
- 已知问题
- 未实现项

### Step 4. 强制切换为 Reviewer

Reviewer 必须覆盖：

#### A. 规格一致性检查

- 是否偏离 `docs/strategy_spec.md`
- 是否偏离 `docs/implementation_plan.md`
- 是否引入新的未定义行为

#### B. 状态与时序检查

- 有没有 lookahead
- 有没有 replay / live / UI 状态错位
- 有没有 order / position orphan 状态
- 有没有同 bar 冲突未定义清楚

#### C. 风控与执行检查

- 风控是否真的先于 intent / order
- pending reservation 是否真的生效
- kill switch 是否真的阻断路径
- 是否可能重复下单
- 是否可能 silent fail

#### D. 前后端边界检查

- 前端有没有偷偷重算后端逻辑
- schema 和 UI 字段是否一致
- router 有没有绕过 engine

#### E. 测试质量检查

- 测试是否只测 happy path
- 有没有覆盖最危险的边界
- 有没有“代码改了但没补对应测试”

### Step 5. Reviewer 判定

判定只能是：

- `PASS`
- `PASS WITH NON-BLOCKING NOTES`
- `FAIL`

规则：

- 只要存在 phase 级别硬问题，必须 `FAIL`
- `FAIL` 后必须回到 Implementer 修复，不得直接宣布完成

## Reviewer 最低审查强度

每次 Reviewer 至少要回答：

1. 这次实现有没有违反 phase 边界？
2. 有没有为了省事把逻辑塞进了不该放的层？
3. 有没有新增“文档没定义但代码自己发明”的行为？
4. 有没有测试没覆盖到的关键边界？
5. 有没有数据状态可能进入脏状态但 UI 看不出来？
6. 有没有会导致未来 live trading 出事故的隐患？
7. 有没有命名、schema、字段层面的漂移？
8. 如果交给另一个工程师接着做，会不会因为定义不清而分叉实现？

Reviewer 必须显式写出：

- 本轮最危险的 1~3 个点
- 它们是不是 blocking
- 是否已经修掉

## 固定输出格式

每个 phase 结束后，最终输出必须用这套结构：

### 阶段名称：

### 本次目标：

### 一、Implementer 结果

- 改动文件：
- 核心实现：
- 运行方式：
- 测试结果：
- 已知限制：

### 二、Reviewer 审查

- 审查范围：
- 发现的问题：
- Blocking issues：
- Non-blocking notes：
- 与 docs 是否一致：
- 是否存在状态一致性风险：
- 是否存在边界未覆盖风险：

### 三、最终判定

- `Verdict: PASS / PASS WITH NON-BLOCKING NOTES / FAIL`

### 四、如果 FAIL

- 修复计划
- 准备补哪些测试
- 修复后将重新审查

### 五、如果 PASS 或 PASS WITH NON-BLOCKING NOTES

- 当前 phase 是否可以封板
- 推荐下一 phase
- 不建议现在做的内容
