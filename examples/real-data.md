# 用真实数据演示多轮实验

建议以 **UCI Online Retail 的客户复购任务**作为主要演示，以 **UCI Bike Sharing 的按时间回归任务**作为补充。
数据来源、署名、许可和实验规则见 [预先固定的方案](../docs/REAL-DATA-PROTOCOL.md)。

## 准备和完整重跑

在项目根目录，先按 README 安装后端依赖。Excel 读取库只用于准备数据，单独安装，不加入插件运行时的锁文件：

```powershell
.venv/Scripts/python.exe -m pip install --target .verification/real-data-deps "openpyxl==3.1.5"
.venv/Scripts/python.exe scripts/prepare-real-data.py --root .verification/real-data-inputs --extra-deps .verification/real-data-deps
```

脚本从 UCI 官方下载，保存源文件哈希、清洗计数、特征/标签时间窗和派生 CSV 哈希。零售 CSV 中的客户编号只用于审计，训练时必须排除。原始数据、派生数据和模型保存在本地忽略目录。

完整对照需要一个安装了相同锁定依赖的独立解释器，用于真正重新训练交付物：

```powershell
.venv/Scripts/python.exe scripts/benchmark-real-data.py --data-root .verification/real-data-inputs --root .verification/my-real-data-run --report .verification/my-real-data-report.json --reproduce-python C:/path/to/independent-env/Scripts/python.exe
```

每次使用新的 `--root`。此命令运行 18 组同配方成对比较、两个随机划分诊断、分支/恢复/防误用检查，以及两组独立交付重训。这里没有调用语言模型；模型质量来自 scikit-learn 算法，流程状态与记录由插件和 MLflow 负责。

## 在 DSH 中演示

把生成的 `retail.csv` 放入 DSH 当前工作区，在插件配置中使用 `executionMode: dsh`，并将记录目录放在该工作区内。可以给 DSH 以下任务：

> 使用工作区的 retail.csv 预测客户下一自然月是否复购，目标列 target。anchor 是预测时点，customer_id 是审计标识，两者都不要当作模型特征。按 anchor 做 time 划分，验证比例 0.2、最终测试比例 0.2，种子 7。先训练 LogisticRegression(max_iter=1000)，再从它分支，完整保留已拟合的数据处理，换成 HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15)。在同一验证集比较 average_precision，先不要看最终测试集。使用固定 request_key，保留每个 run_id。

随后继续：

> 找回刚才的两个实验，确认父实验未被覆盖。按验证集结果选择一个模型，只对它 finalize，然后 export，包含数据。在独立目录运行 reproduce.py，确认它重新训练模型并核对验证集与最终测试指标。

这份数据已经按“过去 90 天特征 → 下个月标签”整理。插件不会从任意原始交易表自动推导这个业务定义；真实使用中由 DSH 编写/修改整理脚本，并审查预测时点和可用信息。

真实 DSH PTC 集成检查也可由脚本重跑（需要本机已构建的 DSH checkout）：

```powershell
$env:DSH_CHECKOUT='E:/dsharness'
$env:MLWB_RETAIL_CSV=(Resolve-Path '.verification/real-data-inputs/retail.csv').Path
node scripts/dsh-real-data-check.mjs
```

它调用 DSH 真正的 PTC、工具注册、Session 和沙箱服务，无 LLM。脚本在临时工作区运行，验证一次程序调用完成多步流程、稳定键重放不新增实验、插件卸载重挂后按 Session 找回实验。它不等同于完整 DSH 进程重启，也不证明减少了用户操作或模型 token。
