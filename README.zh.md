# DSH 机器学习实验工作台

[English](README.md) · [接口说明](docs/API.md) · [实测证据](docs/VALIDATION-v0.4.md) · [成熟项目复用说明](docs/UPSTREAM.md)

这是把 **MLflow + Optuna + scikit-learn 适配到 DeepSeek Harness** 的插件：DSH 负责理解需求、写代码和解释结果，插件负责实验状态与交付。v0.4.0 提供 20 个工具，新增独立最终测试集、后台训练超时、DSH 原生沙箱执行和 PTC 批量流程。[功能成熟度审查](docs/MATURITY.md) 如实列出与大型上游平台的差距。

项目的贡献是 Agent 实验流程适配。实验记录和模型产物使用 MLflow，调参使用 Optuna，训练、预处理和指标使用 scikit-learn，没有另写训练算法或调参算法。

## 能解决什么

| 用户操作 | 实际行为 |
|---|---|
| “训练复购模型，比较几个方案” | 每个方案得到独立 MLflow run，绑定数据、代码、参数、环境和固定验证集 |
| “保留第三次实验的数据处理，换一个更快的模型” | 新建分支，复制第三次已拟合的预处理器和验证集索引，父实验保持不变 |
| 关闭 DSH 后再继续 | local 模式继续训练；dsh 模式停止受管进程，重开后标记为 interrupted，保留原记录并分支重训 |
| 比较不同条件的分数 | 数据、目标、验证行或评测契约不一致时拒绝排名 |
| 交付模型和报告 | 核对文件哈希与 MLflow 记录，导出模型、训练代码、报告、依赖及独立复现脚本 |
| “先检查数据，避免同一客户跑到训练集和验证集两边” | 检查缺失值、重复行和疑似 ID；显式配置分组划分；也支持按时间留出较晚记录 |
| “在固定预算内调参数，中断后继续” | Optuna 持久化 study，每个 trial 对应 MLflow run，重复推进不会重复启动同一训练 |
| “用这个模型预测新的 CSV” | 验证输入特征，输出预测、类别概率及校验清单 |

## 安装

需要 Python 3.11+、Node 24.9+、DSH 0.2.2。实测环境为 Windows / Python 3.12.0 / DSH `0.2.2-alpha.3`。其他操作系统暂未实测。

从 [GitHub Release](https://github.com/shinerain1/dsh-ml-workbench/releases) 下载 `.whl` 后端、`requirements.lock` 与 `.tgz` 插件。先启动过一次 DSH Desktop 以初始化配置，再完全退出它，然后在专用目录运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock .\dsh_ml_workbench-0.4.0-py3-none-any.whl
dsh plugin --profile desktop add .\dsh-ml-workbench-0.4.0.tgz
```

也可打开 DSH **插件 → 实验性插件 → 自定义插件索引**，进入 **发现 → 添加索引源**，填写：

```text
https://raw.githubusercontent.com/shinerain1/dsh-ml-workbench/main/market/index.json
```

这是第三方发布索引，Git 源固定到具体 commit，包含编译产物。DSH 当前不内置统一索引源；发布此索引不表示官方审核或背书。通过“发现”安装插件后，仍需安装 Python 后端。

在插件配置中设置 Python 解释器和记录目录的**绝对路径**，例如：

```yaml
- id: ml-workbench
  config:
    pythonExecutable: C:/ml-workbench/.venv/Scripts/python.exe
    root: C:/my-project/.dsh-ml-workbench
    executionMode: dsh
    allowCustomCode: false
```

先让 DSH 调用 `ml_workbench_health` 确认后端可用。重开 DSH 时使用同一个 `root` 即可找回实验。数据和 MLflow 数据库保存在本机；插件不上传用户数据。

也可从源码目录或解压后的插件 `package` 目录运行安装助手：

```powershell
py -3.12 scripts/install.py --venv C:/ml-workbench/.venv --workspace C:/my-project --wheel C:/downloads/dsh_ml_workbench-0.4.0-py3-none-any.whl
```

助手安装锁定依赖、检查版本和 SQLite 写入，生成 `ml-workbench.config.json`，不会覆盖 DSH 配置。将生成的配置应用到插件。dsh 模式要求记录目录位于当前 DSH 工作区或已经批准的额外可写目录内，不会自动扩大权限。独立自检命令：`python -m dsh_ml_workbench doctor --root C:/my-project/.dsh-ml-workbench`。

## 使用流程

1. 提供 CSV 绝对路径、标签列和任务类型；调用 `ml_dataset_inspect` 检查数据。排除 ID、结果发生后的字段等泄漏列，并明确选择随机、分组或时间划分。
2. 调用 `ml_experiment_run` 一步固定输入并启动，也可保留 create/start 两步。用 `ml_experiment_wait` 有界等待；不能把 starting 当成完成。[PTC 示例](docs/PTC-WORKFLOW.md) 将多步操作组合为一次程序调用。
3. 多方案完成后，用 `ml_experiment_diff` 查看变化、`ml_experiment_compare` 比较指标。分类还支持适用情况下的 ROC-AUC、平均精确率和 log loss；报告包含简单基线与混淆矩阵。
4. 保留某次数据处理时，`ml_experiment_branch` 默认复制其**已经拟合的预处理器**。新模型要同时覆盖 `params`，防止把森林参数传给线性模型。
5. 新实验默认训练 60%、验证 20%、最终测试 20%（分组/时间边界可能调整行数）。只用验证集选模型，选定后调用一次 `ml_experiment_finalize`，再 export。复现脚本重新拟合模型并分别核对验证、最终测试成绩。CSV 默认不包含，需要时设置 `include_data: true`。
6. 需要调参时，用 `ml_search_create` 定义预算和搜索空间，重复调用 `ml_search_step` 推进，使用 `get/cancel` 查询或停止。详见 [Optuna 调参契约](docs/TUNING.md)。
7. 用 `ml_model_predict` 对新 CSV 批量预测；实验列表支持状态、模型、父实验过滤和分页。

创建参数示例：

```json
{"dataset_path":"C:/data/customers.csv","target":"repurchased","task":"classification","model":"random_forest","params":{"n_estimators":100},"preprocessing":{"drop_columns":["customer_id"]},"request_key":"customers-forest-001"}
```

复用第三次实验的示例：

```json
{"run_id":"第三次实验返回的ID","changes":{"name":"保留第三次预处理，改用逻辑回归","model":"logistic_regression","params":{}},"reuse_preprocessing":true,"request_key":"customers-linear-004"}
```

`request_key` 用于同一创建请求的幂等重试；改动实验要使用新 key。`start` 重复调用不会重复启动训练。若要修改预处理、划分或从失败实验重新训练，设置 `reuse_preprocessing: false`。首版支持从快照重启训练，不支持训练中间权重恢复。

重复客户数据可设置 `split: {"strategy":"group","column":"customer_id"}`；时间数据可设置 `split: {"strategy":"time","column":"event_date"}`。划分列自动排除出特征，时间列要求 ISO 日期/时间字符串，相同时间戳不会跨越两边。时间划分本身不等于完整的时序预测方法。

从 v0.1.0 升级时保留原记录目录。旧实验可以查询、比较和导出；复用旧预处理器的分支会保留旧训练配方。要使用新版评测指标或调参，请用 `reuse_preprocessing:false` 新建分支。不同评测版本不会直接混排分数。

## 交付与复现

解压 ZIP，在独立虚拟环境内：

```sh
python -m pip install -r inputs/requirements.txt
python reproduce.py --verify-only
python reproduce.py --data /path/to/original.csv
```

含数据的包不需要 `--data`。脚本验证数据哈希、依赖版本和文件校验值，再重新训练并比较指标；不依赖原工作台数据库。模型可以通过 `mlflow.sklearn.load_model("output/model")` 加载，包含预测所需预处理。

`ml_workbench_health` 返回 MLflow `tracking_uri`，可用 `mlflow ui --backend-store-uri <tracking_uri> --host 127.0.0.1` 查看标准 MLflow 界面。不要直接改记录或快照文件，否则导出一致性检查会拒绝交付。

## 边界与实测

自定义 Python 工厂示例位于 [examples](examples)。`allowCustomCode` 默认关闭。`executionMode: dsh` 将控制请求和训练交给 DSH 的 sandbox、sandboxPolicy、subprocess 服务，无服务或无法约束时拒绝执行；local 模式保留主机后台执行。Windows 上 DSH 报告部分文件约束，不能将其描述为禁止读取、断网或恶意代码隔离。仅支持本地、单用户、可信工厂和可信模型，尚不支持 SSH 后端。

默认采用 IID 随机留出法，并提供分组/时间划分。未实现并行调参、剪枝、嵌套交叉验证、分布式训练、DVC 或 RD-Agent 接入。仍需 DSH 调用 step 推进下一轮；单次训练超时由独立监护进程执行，不需要轮询。最终测试分区对一个模型锁定，属于本地流程约束，不能阻止查看源 CSV，也不能自动判断业务泄漏。旧实验分区不变；`final_test_fraction: 0` 显式关闭最终测试集。

[验证报告](docs/VALIDATION-v0.4.md) 区分确定性集成测试与真实 Agent 对照实验。已运行的合成数据结果不能被写成真实复购业务收益，也不能推导“减少人工介入”的结论。

新增 DSH 会话关联、同一调用的自动去重和 `ml_workbench_resume` 恢复工具。详见 [DSH 适配说明](docs/DSH-INTEGRATION.md)和[对照实验报告](docs/BENCHMARK.md)。

新增[真实数据实验报告](docs/REAL-DATA-EXPERIMENTS.md)：UCI 客户复购与单车需求两组任务，检验多轮分支、比较约束、恢复和交付复现。[复跑命令与 DSH 演示提示词](examples/real-data.md)可直接使用。这里验证的是流程保障，不等于更高预测准确率或更少人工介入。
