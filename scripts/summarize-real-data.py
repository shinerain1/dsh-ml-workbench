"""Validate completed experiment evidence and render its Chinese report."""
import json
import statistics
from pathlib import Path

project = Path(__file__).resolve().parents[1]
evidence = json.loads((project / "docs/real-data-results.json").read_text(encoding="utf-8"))
ptc = json.loads((project / "docs/dsh-real-data-results.json").read_text(encoding="utf-8"))
assert len(evidence["datasets"]) == 2
assert ptc["context_disposed"] and ptc["plugin_remount_session_recovered"]
tables, final_rows, timing_rows, diagnostic_rows = [], [], [], []
names = {"retail": "客户复购", "bike": "单车小时需求"}
model_names = {"logistic_regression": "逻辑回归", "ridge": "Ridge", "random_forest": "随机森林", "hist_gradient_boosting": "直方图梯度提升"}
for data in evidence["datasets"]:
    assert len(data["pairs"]) == 9 and len(data["branch_checks"]) == 6
    assert all(p["maximum_absolute_metric_delta"] == 0 and p["plugin_record_categories"] == p["direct_record_categories"] == 8 for p in data["pairs"])
    assert len(data["workflow_checks"]) == 7 and all(x["passed"] for x in data["workflow_checks"].values())
    assert all(data["direct_explicit_controls"].values())
    assert all(data["direct_independent_retraining"].values())
    assert all(x["parent_outputs_unchanged"] and x["fitted_preprocessor_identical"] for x in data["branch_checks"])
    classification = data["task"] == "classification"
    metric = data["selection_metric"]
    metric_name = "AP（越高越好）" if classification else "RMSE（越低越好）"
    selected = next(p for p in data["pairs"] if p["run_id"] == data["selected_run_id"])
    table = [f"### {names[data['name']]}", "", f"| 模型 | 验证集 {metric_name}，3 种子均值 | 最小—最大 | 拟合时间中位数 |", "|---|---:|---:|---:|"]
    for model in dict.fromkeys(p["model"] for p in data["pairs"]):
        rows = [p for p in data["pairs"] if p["model"] == model]
        values = [p["metrics"][metric] for p in rows]
        table.append(f"| {model_names[model]} | {statistics.mean(values):.4f} | {min(values):.4f}—{max(values):.4f} | {statistics.median(p['plugin_fit_seconds'] for p in rows):.3f} 秒 |")
    table += [f"| 常数预测基线 | {selected['dummy_metrics'][metric]:.4f} | — | — |", "", "实际时间分区：", "", "| 分区 | 样本数 | 起点 | 终点 |", "|---|---:|---|---|"]
    for key, label in [("train", "训练"), ("test", "验证"), ("final_test", "最终测试")]:
        p = data["partitions"][key]
        table.append(f"| {label} | {p['rows']:,} | {p['start']} | {p['end']} |")
    tables.append("\n".join(table))
    secondary = "roc_auc" if classification else "mae"
    final_rows.append(f"| {names[data['name']]} | {model_names[selected['model']]} / {selected['seed']} | {selected['metrics'][metric]:.4f} | {data['final_test']['metrics'][metric]:.4f} | {secondary}={data['final_test']['metrics'][secondary]:.4f} |")
    time_rf = next(p for p in data["pairs"] if p["model"] == "random_forest" and p["seed"] == 7)
    diagnostic_rows.append(f"| {names[data['name']]} | {metric} | {time_rf['metrics'][metric]:.4f} | {data['random_split_diagnostic']['validation_metrics'][metric]:.4f} | 拒绝混排 |")
    timing_rows.append(f"| {names[data['name']]} | {statistics.median(p['plugin_workflow_seconds'] for p in data['pairs']):.2f} 秒 | {statistics.median(p['direct_workflow_seconds'] for p in data['pairs']):.2f} 秒 |")

report = f"""# 真实数据实验：什么任务能体现 DSH ML Workbench 的价值

日期：2026-11-10；插件/后端 v0.4.0；Windows；DSH 0.2.2-alpha.3。

**优先使用客户复购预测演示多轮实验，再用单车时间回归演示评测条件保护。**
两组真实数据的 18 组配对实验已经完成。插件与直接 MLflow 使用同样的训练算法时，
全部指标相同；插件的可验证贡献是把分支、恢复、比较和交付检查变成 DSH 可以直接调用的流程。
本次没有新的自然语言 Agent A/B 试验，不能据此声称节省人工或 token。

## 为什么选择这两组数据

| 任务 | 数据与业务问题 | 适合展示的功能 |
|---|---|---|
| 客户复购分类 | [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)：过去 90 天有购买的客户，下个自然月是否购买 | 业务特征处理、同一预处理换模型、记录父子关系、跨轮找回、完整交付 |
| 单车需求回归 | [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)：给定日历和实际天气，估计小时租车量 | 时间划分、评测条件核对、分类之外的回归工作流、独立最终测试 |

Online Retail 由 Chen（2015）提供，DOI 10.24432/C5BW33；Bike Sharing 由 Fanaee-T（2013）提供，
DOI 10.24432/C5W894；均为 CC BY 4.0。数据转换由本项目脚本完成，原始来源与派生数据哈希见结果 JSON。

零售原表 541,909 行，依次移除缺失客户号 135,080 行、取消记录 8,905 行、非正数量/价格 40 行、
完整重复记录 5,192 行，剩余 392,692 行。构建 2011 年 3—11 月的 **17,837 条客户月度样本**，
包含距最近购买的天数、订单数、金额、数量、商品种类、平均订单金额六项历史特征。
客户编号和 anchor 不作为预测特征。每个标签只使用 anchor 起的下一个自然月；检查了特征截止时间
和分区交界处的标签成熟时间。这是近期活跃且有编号客户的复购任务，不代表所有访客或新客户。

单车使用 hour.csv 的全部 **17,379 行**。排除 `casual`、`registered`，因为两者相加就是目标 `cnt`；
排除行号和原始日期，用构造的时间戳划分。季节、月份、小时、星期和天气类别按类别编码。
使用了当时的实际天气，因此不能把本实验称为可部署的提前需求预测系统。

## 对照规则

[方案](REAL-DATA-PROTOCOL.md)在查看模型结果前固定。每个任务使用 3 类模型 × 种子 7、19、41，
申请 60/20/20 的时间分区，同一时间点不能拆开。三个种子共享同一分区，它们不是三份独立数据样本。
数值预处理仅在训练集拟合；分支复制已经拟合的处理器。类别字段使用默认缺失填充和独热编码。
分类按验证集 AP（平均精确率）选模型，回归按验证集 RMSE 选模型，然后仅评估选定模型的最终测试集。

直接 MLflow 对照获得相同的冻结代码、数据和分区，独立拟合模型，并用原生 API 完整记录产物。
这是有利于对照组的共享配方设计，用来隔离流程机制；不是让两个 LLM 独立编写代码的对比。
另外提供显式 key、分区哈希、产物哈希和选择标签检查，证明熟练用户也可以在 MLflow 上编程实现这些保护。

## 实际模型结果

{(chr(10) * 2).join(tables)}

同时间点保护使零售实际分区偏离 60/20/20，这是预期行为。仅改模型种子不会改变这份时间分区。
当前小规模设定下，部分算法在三个种子上分数相同；不把这种一致性当作统计显著性证据。

选定模型的独立最终测试如下。两个任务指标单位不同，不能横向比较数值大小。

| 任务 | 验证集选定模型 / 种子 | 验证 AP 或 RMSE | 最终测试 AP 或 RMSE | 最终测试补充指标 |
|---|---|---:|---:|---|
{chr(10).join(final_rows)}

复购最终测试的 ROC-AUC 约 0.68，效果有提升空间；最终时段的客户复购率也变了。
这些分数不等于营销收益，尚未建立触达成本、收益阈值或未来新时期的业务验证。

## 哪些优势被实际验证

| 检查 | 结果 | 能支持的结论 |
|---|---|---|
| 同配方插件 / 直接 MLflow | 18/18 组指标最大绝对差为 0 | 插件保留成熟算法结果；没有凭空增加预测能力 |
| 记录完整性 | 两组每次均 8/8 类 | 仔细编写的 MLflow 脚本也能完整记录，插件提供默认流程 |
| 分支复用与父实验保护 | 12/12 分支复制的预处理哈希相同，父输出未变 | 换模型时可以保留数据处理和旧成果 |
| 两个数据集各 7 项流程检查 | 14/14 通过 | 重放、恢复、比较约束、最终测试约束和交付检查在这两组数据上生效 |
| 独立解释器从交付目录重新拟合 | 插件 2/2、直接配方 2/2；验证和最终测试都匹配 | 两者均能复现；插件直接生成交付包 |
| 真实 DSH PTC / Session / sandbox | 完整流程 1 次外层调用、{ptc['nested_calls_for_workflow']} 次嵌套工具调用；重放 ID 不变；重挂插件找回 2 个实验 | DSH 原生工具组合、会话关联和工作区执行确实可用 |

7 项流程检查分别是：拒绝时间/随机分区混排、拒绝同一最终分区选第二个模型、拒绝导出被改动的模型、
后端重开后按 Session 找回、稳定 key 恢复丢失响应、重复 finalize 返回同一结果、独立交付重训匹配。
前述计数是预先指定的注入/验证用例，不能解释为生产错误率下降 100%。

### 评测条件变化确实会改变分数

额外只对 seed=7 随机森林改变划分方式，未查看随机划分的最终测试集：

| 任务 | 指标 | 时间划分验证 | 随机划分验证 | 插件行为 |
|---|---|---:|---:|---|
{chr(10).join(diagnostic_rows)}

零售的随机划分分数反而较低，不能预设随机划分总会“虚高”。同一客户的不同月份落在时间分区两侧，
对“预测老客户未来复购”是合理设定；预测未见过的新客户则需要另设分组评估。单车随机划分和时间划分
回答的泛化问题不同。插件拒绝混排的价值是不让 Agent 把不同条件的分数直接当成模型优劣。
插件并不自动选择正确划分，也不自动识别标签泄漏。

### 也记录了成本和限制

本机每次创建/分支、训练到记录完成的流程耗时中位数：

| 任务 | 插件 | 同进程直接 MLflow |
|---|---:|---:|
{chr(10).join(timing_rows)}

插件要启动独立监护/训练进程、进行校验和持久化，当前短任务的流程开销明显更大。
这不是公平的纯训练速度基准：两边执行架构不同、固定串行顺序且存在缓存影响；表中数据不能推出通用速度倍率。
本次没有证明“插件更快”，也没有测量用户人工介入或模型 token。先前真实 Agent 小样本对照中的不利结果仍保留在
[BENCHMARK.md](BENCHMARK.md)，不能用本次无 LLM 测试覆盖它。

DSH 集成使用真实 workspace-write policy、sandboxPolicy/subprocess 服务和 PTC runtime。
插件卸载重挂继续使用同一个 DSH Session，不等于完整 DSH 进程重启。独立集成工作区重复固定流程，
不作为另一份独立统计验证。Windows 提供的文件约束也不代表断网或禁止读取所有工作区外文件。

## 开发中发现的问题与建议

本次未发现需要改动 v0.4.0 后端的训练/比较/导出缺陷。新编写的实验辅助脚本有两处问题已修正：
嵌套 ZIP 中直接读取 XLSX 发生 seek/EOF 错误，改为在内存中展开内层文件；DSH 检查脚本使用了错误的
Context 清理方法，改用 `ctx.fiber.dispose()`。首轮 DSH 工作流已成功但进程收尾失败，修正后在新工作区
完整重跑且退出码为 0。没有因此改变数据、模型参数、指标或成功判据。

后续最值得做的是：缩短短任务的 Python/MLflow 启动开销；增加滚动时间窗口与 gap/embargo，验证更多未来时段；
再做多任务、交叉顺序的真实 DSH Agent 对照，测量完成率、工具调用、token 和人工介入。
现阶段最适合将插件定位为“让 DSH 的多轮机器学习实验更可核对、更容易继续和交付”。

## 复跑与原始证据

- [数据准备、完整对照命令和 DSH 演示提示词](../examples/real-data.md)
- [真实数据结果 JSON](real-data-results.json)：全部 18 组分数、耗时、数据哈希、分区、检查结果。
- [真实 DSH 集成结果 JSON](dsh-real-data-results.json)：PTC、重放、Session 和清理证据。
- [实验脚本](../scripts/benchmark-real-data.py)、[数据脚本](../scripts/prepare-real-data.py)、[DSH 脚本](../scripts/dsh-real-data-check.mjs)。

原始交易、客户行、训练数据库和含数据模型包仅留本地。公开仓库发布脚本和汇总证据。
"""
(project / "docs/REAL-DATA-EXPERIMENTS.md").write_text(report, encoding="utf-8")
print(json.dumps({"pairs": 18, "workflow_checks": 14, "branch_checks": 12, "report": "docs/REAL-DATA-EXPERIMENTS.md"}))
