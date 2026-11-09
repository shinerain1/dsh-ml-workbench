# DSH 机器学习实验工作台

[English](README.md) · [接口说明](docs/API.md) · [实测证据](docs/VALIDATION.md) · [成熟项目复用说明](docs/UPSTREAM.md)

这是把 **MLflow + scikit-learn 适配到 DeepSeek Harness** 的插件：DSH 负责理解需求、写代码和解释结果，插件负责可靠地创建、启动、查询、取消、比较、分支和导出实验。首版支持本地 CSV 分类与回归。

项目的贡献是 Agent 实验流程适配。实验记录和模型产物使用 MLflow，训练、预处理和指标使用 scikit-learn，没有另写训练算法。

## 能解决什么

| 用户操作 | 实际行为 |
|---|---|
| “训练复购模型，比较几个方案” | 每个方案得到独立 MLflow run，绑定数据、代码、参数、环境和固定验证集 |
| “保留第三次实验的数据处理，换一个更快的模型” | 新建分支，复制第三次已拟合的预处理器和验证集索引，父实验保持不变 |
| 关闭 DSH 后再继续 | 后台训练继续；重新查询持久化状态；进程异常退出标记为 interrupted |
| 比较不同条件的分数 | 数据、目标、验证行或评测契约不一致时拒绝排名 |
| 交付模型和报告 | 核对文件哈希与 MLflow 记录，导出模型、训练代码、报告、依赖及独立复现脚本 |

## 安装

需要 Python 3.11+、Node 24.9+、DSH 0.2.2。实测环境为 Windows / Python 3.12.0 / DSH `0.2.2-alpha.3`。其他操作系统暂未实测。

从 [GitHub Release](https://github.com/shinerain1/dsh-ml-workbench/releases) 下载 `.whl` 后端、`requirements.lock` 与 `.tgz` 插件。先启动过一次 DSH Desktop 以初始化配置，再完全退出它，然后在专用目录运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock .\dsh_ml_workbench-0.1.0-py3-none-any.whl
dsh plugin --profile desktop add .\dsh-ml-workbench-0.1.0.tgz
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
    root: C:/ml-workbench/records
    allowCustomCode: false
```

先让 DSH 调用 `ml_workbench_health` 确认后端可用。重开 DSH 时使用同一个 `root` 即可找回实验。数据和 MLflow 数据库保存在本机；插件不上传用户数据。

## 使用流程

1. 提供 CSV 绝对路径、标签列和任务类型；让 DSH 排除 ID、结果发生后的字段等泄漏列。
2. 调用 `ml_experiment_create` 固定输入，再 `ml_experiment_start` 启动。训练异步运行，通过 `ml_experiment_get` 查询，不能把 `starting` 当成完成。
3. 多方案完成后，用 `ml_experiment_compare` 比较。默认分类 accuracy 越高越好，回归 RMSE 越低越好。
4. 保留某次数据处理时，`ml_experiment_branch` 默认复制其**已经拟合的预处理器**。新模型要同时覆盖 `params`，防止把森林参数传给线性模型。
5. `ml_experiment_export` 生成本地 ZIP，默认不包含 CSV。需要自包含数据交付时显式设置 `include_data: true`。

创建参数示例：

```json
{"dataset_path":"C:/data/customers.csv","target":"repurchased","task":"classification","model":"random_forest","params":{"n_estimators":100},"preprocessing":{"drop_columns":["customer_id"]},"request_key":"customers-forest-001"}
```

复用第三次实验的示例：

```json
{"run_id":"第三次实验返回的ID","changes":{"name":"保留第三次预处理，改用逻辑回归","model":"logistic_regression","params":{}},"reuse_preprocessing":true,"request_key":"customers-linear-004"}
```

`request_key` 用于同一创建请求的幂等重试；改动实验要使用新 key。`start` 重复调用不会重复启动训练。若要修改预处理、划分或从失败实验重新训练，设置 `reuse_preprocessing: false`。首版支持从快照重启训练，不支持训练中间权重恢复。

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

自定义 Python 工厂示例位于 [examples](examples)。`allowCustomCode` 默认关闭；开启后代码以当前用户权限在 DSH 主机执行，不属于 DSH 的 SSH/进程沙箱。只执行审阅过的工厂、加载可信模型；首版面向单个本地用户。

首版采用 IID 随机留出法（分类分层抽样），不适用于未经处理的时间序列或有分组泄漏的数据。不包含自动调参、分布式训练、DVC 或 RD-Agent 接入。多轮挑选同一验证集也会带来选择偏差，最终效果应使用独立测试集评估。

[验证报告](docs/VALIDATION.md) 区分确定性集成测试与真实 Agent 对照实验。已运行的合成数据结果不能被写成真实复购业务收益，也不能推导“减少人工介入”的结论。
