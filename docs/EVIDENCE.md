# 功能声明与验证证据

本页用于维护者或用户核对插件是否做到了说明中的功能。结果不是仅凭 README 自述：
对应的执行脚本、测试输出、数据哈希与完整指标均在仓库中，可以复跑和检查。

**证据范围**：v0.4.0，Windows / Python 3.12.0，DSH 0.2.2-alpha.3。
发布源码固定于 [`86d2bb6`](https://github.com/shinerain1/dsh-ml-workbench/tree/86d2bb6602aa5d33ccbac5a6318ea22061ede970)。
真实数据脚本与汇总首个提交为 [`158573f`](https://github.com/shinerain1/dsh-ml-workbench/commit/158573f20f8f34b6b19aebf3ddda4f51eb35339a)。

| 声明 | 实际怎样验证 | 证据与实现 |
|---|---|---|
| 多轮换模型可保留旧成果 | 两个真实任务共 12 个分支；复制的已拟合预处理器哈希相同；父输出文件哈希不变 | [结果 JSON](real-data-results.json)中的 `branch_checks`；[实验脚本](../scripts/benchmark-real-data.py)；[branch 实现](../python/dsh_ml_workbench/service.py) |
| 避免不同条件的分数直接混排 | 两个任务各制造时间/随机分区混用，比较返回 `INCOMPARABLE_EXPERIMENTS` | 结果中的 `workflow_checks.mixed_split_comparison`；service.py 的 `compare` 与快照契约 |
| 减少把最终测试当调参集的误用 | 先用验证集选一个模型并 finalize，再尝试 finalize 同分区另一个模型，被拒绝；同模型重试结果不变 | `second_final_selection`、`same_final_selection_replay`；service.py 的 `finalize` |
| 请求重放与会话找回 | 稳定请求键返回同一个 run；重新创建后端对象可按 Session 找回；真实 DSH 插件重挂后能找回实验 | `lost_response_stable_key`、`backend_reopen_session_navigation`；[真实 DSH 结果](dsh-real-data-results.json) |
| 不把被改动的模型作为正常交付导出 | 成功训练后给模型文件追加字节，export 返回 `INTEGRITY_MISMATCH`；恢复原字节后可正常导出 | `workflow_checks.mutated_model_export`；service.py 的 `export` 与文件校验 |
| 交付物能独立重新训练 | 两份真实数据导出包在独立安装的解释器中重新拟合模型，验证与最终测试指标均匹配 | `independent_standalone_retraining`；[独立复现脚本](../python/dsh_ml_workbench/reproduce.py) |
| 实际使用 DSH 原生能力 | 真实 PTC runtime、Session、sandboxPolicy 和 subprocess 下，一次外层调用组合 8 次工具操作，完成训练到导出；重放 ID 不变 | [脚本](../scripts/dsh-real-data-check.mjs)、[结果](dsh-real-data-results.json)，不是伪造 runtime |
| 可从发布产物安装 | 验证 Release 五份产物摘要、公开 Git 安装、实际 DSH profile 的依赖解析、20 个工具加载/健康检查/卸载 | [发布验证](publication-v4.json)、[打包验证](packaging-v4-results.json)、[v0.4 总报告](VALIDATION-v0.4.md) |

## 实验规模与对照是否公平

两组公开数据：UCI Online Retail 客户复购分类（17,837 条客户月度样本）与 Bike Sharing 回归（17,379 条小时记录）。
每组 3 类模型 × 3 个种子，插件和直接 MLflow 接收相同冻结配方与数据，总计 **18 组成对比较**。
两边所有记录指标完全相同，8 类记录均齐全；两边都能独立重训。
这 8 类是数据、配置、分区、训练配方、环境、依赖、指标和模型；完整性计数检查它们是否存在，文件一致性由单独的哈希检查验证。

直接 MLflow 的显式请求键查询、分区哈希、模型哈希与选择标签检查也通过。
因此贡献是把这些流程保障适配为 Agent 可调用的默认行为，不能宣称 MLflow 本身做不到。
随机划分的分数不总更高，插件也有额外启动开销；不利结果同样保留。

[预先固定的实验方案](REAL-DATA-PROTOCOL.md)的 SHA-256 记录在结果 JSON，
[完整报告](REAL-DATA-EXPERIMENTS.md)包含所有候选模型、实际时间分区、最终测试、耗时和限制。
公开数据来源、许可、清洗计数、特征窗口和源/派生文件哈希也在 JSON 中。

## 怎样复核

1. 按 [复跑说明](../examples/real-data.md)下载原始数据并重建 CSV。
2. 用新的实验目录运行 `scripts/benchmark-real-data.py`，为交付重训指定独立解释器。
3. 用构建好的 DSH checkout 运行 `scripts/dsh-real-data-check.mjs`，核对实际 DSH 工具链。
4. 阅读输出 JSON；检查预期错误确实发生、复现脚本真的调用训练，而非仅加载原模型重打分。

源数据、客户行和含数据模型包不随仓库发布。报告提供重新获取公开数据与生成结果的方法。

## 不能从这些结果推出什么

- 14/14 是预设检查用例通过，不是生产环境错误率下降 100%。
- 真实 DSH 集成测试没有 LLM 调用；插件重挂也不是完整 DSH 进程重启。
- 当前真实数据实验未证明人工介入或 token 减少。此前小样本真实 Agent 对照中的更多工具调用和参数错误保留在 [BENCHMARK.md](BENCHMARK.md)。
- 没有证明业务 ROI、统计显著的预测优势、跨系统成熟度或对不可信代码的完整隔离。
- 最终测试选择锁是本地工作流约束，不能阻止使用者另建工作区或直接读取源 CSV。

这组证据支持的表述是：**在已验证任务中，插件落实了可复查的实验分支、比较限制、恢复和交付检查。**
