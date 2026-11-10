# DSH 机器学习实验工作台：安装与使用

**让 DSH 的多轮机器学习实验可追踪、可比较、可继续迭代，并能复现交付。**
插件把 MLflow、Optuna 和 scikit-learn 接到 DSH 工具上。第一版范围是本地 CSV 分类与回归，面向单个可信用户。

例如：“训练客户复购模型，比较几个方案。”之后继续：“保留第二次实验的已拟合数据处理，换一个模型。”
插件为新方案建立独立实验，绑定数据、代码、参数、分区和产物，保留父实验。

| 能做什么 | 对应行为 |
|---|---|
| 多方案与分支 | 创建、启动、查询、取消；复制父实验已拟合的预处理器并训练新模型 |
| 继续实验 | 持久化实验状态；按 DSH Session 找回实验；稳定请求键避免重复创建 |
| 合理比较 | 不同数据或验证分区拒绝混排；用验证集选定模型后再评估最终测试集 |
| 调参和预测 | 持久化 Optuna 顺序试验；使用已训练模型预测新的本地 CSV |
| 核对交付 | 校验模型与记录，导出模型、代码、报告、依赖和独立重新训练脚本 |
| DSH 集成 | 原生工具、PTC 程序组合、Session 关联，以及可选 DSH 沙箱执行 |

## 安装需要两部分

**DSH 插件安装不包含 Python 环境。** 需要安装同版本 Python 后端，再配置插件。

已验证版本：插件/后端 **0.4.0**；DSH **0.2.2-alpha.3**；Windows；Python **3.12.0**。
声明兼容范围为 Python ≥3.11、Node ≥24.9、DSH `>=0.2.2-alpha.3 <0.3.0`。
该范围不表示每个版本和操作系统都已实测。

### 1. 安装 Python 后端

从 [v0.4.0 发布页](https://github.com/shinerain1/dsh-ml-workbench/releases/tag/v0.4.0)下载
`requirements.lock` 和 `dsh_ml_workbench-0.4.0-py3-none-any.whl`。
在专用目录打开 PowerShell：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock ./dsh_ml_workbench-0.4.0-py3-none-any.whl
```

也可用仓库的 [安装助手](../scripts/install.py)安装匹配版本、检查环境并生成配置示例。

### 2. 安装 DSH 插件

使用 DSH 的插件安装功能，安装这个预构建包：

```text
https://github.com/shinerain1/dsh-ml-workbench/releases/download/v0.4.0/dsh-ml-workbench-0.4.0.tgz
```

对于自行维护的 `web` profile，可执行：

```powershell
dsh plugin --profile web add https://github.com/shinerain1/dsh-ml-workbench/releases/download/v0.4.0/dsh-ml-workbench-0.4.0.tgz
```

Desktop 用户应在自己实际使用的桌面应用插件界面安装；`--profile web` 不会安装到 Desktop profile。

也可以在 DSH 的“自定义插件索引 → 发现”中添加本项目已经公开的独立索引：

```text
https://raw.githubusercontent.com/shinerain1/dsh-ml-workbench/main/market/index.json
```

这是项目自行发布的安装源。社区广场的收录状态与它分开，[投稿材料](../market/submission/README.md)不代表已经获准上架。

### 3. 配置并检查

在插件设置中填写后端 Python 与记录目录的**绝对路径**。例如当前 DSH 工作区为 `C:/my-project`：

```yaml
- id: ml-workbench
  config:
    pythonExecutable: C:/ml-workbench/.venv/Scripts/python.exe
    root: C:/my-project/.dsh-ml-workbench
    executionMode: dsh
    allowCustomCode: false
```

`root` 应在当前 DSH 工作区或已批准的可写目录内，路径需改成你自己的实际位置。
先让 DSH 调用 `ml_workbench_health`。确认版本和解释器正确，再开始训练。

配置说明：`executionMode: dsh` 使用 DSH 的执行约束；插件/DSH 退出会停止它拥有的训练进程。
`local` 是插件的兼容默认值，使用主机后台执行，可在 DSH 关闭后继续训练；安装助手生成 `dsh` 配置。
恢复未完成训练会保留原记录并分支重新训练，不支持模型权重断点续训。

## 可以直接发给 DSH 的任务

> 使用当前工作区的 customers.csv，目标列 repurchased，做客户复购分类。先检查数据，排除 customer_id 和预测时点之后才出现的信息，确定合适的时间或分组划分。比较逻辑回归和随机森林，保留相同的验证与最终测试分区。为每次实验记录 run_id，暂不查看最终测试结果。

> 从第二次实验新建分支，复用它已经拟合的预处理器，换成另一个模型，同时替换模型参数。只在相同评测条件下比较。选定后 finalize，再导出含代码、模型和复现脚本的包，实际重新训练并核对指标。

可复跑的公开数据例子见 [真实数据演示](../examples/real-data.md)。

## 有哪些证据，仍有哪些限制

[证据入口](EVIDENCE.md)将每项声明对应到代码、测试和机器可读结果。
真实数据实验验证了 12 个分支、14 项预先定义的流程检查和两份独立交付重训；
18 组成对实验与直接 MLflow 指标相同。
这些结果支持具体流程行为，不代表自动提高模型准确率、减少人工介入或达到大型平台的生产成熟度。

当前不提供独立图形实验面板；查看图表可使用 MLflow 自身界面。尚无 GPU/分布式训练、自动业务泄漏识别、
滚动时间交叉验证、DVC 数据远程管理或模型在线部署。短任务存在额外启动和校验开销。
Windows 的 DSH 约束不是完整网络或读取隔离。自定义 Python 工厂需要显式启用；仅加载可信模型。

数据、模型与追踪库保存在本地。`include_data` 默认关闭；需要将 CSV 放入导出包时再显式开启。
MIT 许可适用于本适配器；MLflow、Optuna、scikit-learn 保持各自上游许可。
