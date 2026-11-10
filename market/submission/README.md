# 社区插件广场投稿材料

当前状态：**材料已准备，尚未提交广场收录 PR。**

投稿目标为 [awesome-dsh-plugin/awesome-dsh-plugin](https://github.com/awesome-dsh-plugin/awesome-dsh-plugin)。
[dsh-market 的投稿说明](https://github.com/dsh-market/dsh-market#submit-your-plugin)将收录导向这个目录；
它要求一个插件对应一个 YAML 文件，而不是把 Python 后端上传给市场运行。

## 提交时使用

1. 在目录仓库的 fork 中新建分支。
2. 把本目录的 `shinerain1__dsh-ml-workbench.yml` 原样放到目录仓库的 `data/plugins/` 下。
3. 只提交这一个新增文件，不改生成的 README 或其他插件条目。
4. 创建 PR，建议标题：`Add shinerain1/dsh-ml-workbench`；正文使用 [PR_BODY.md](PR_BODY.md)。

提交前再核对目录的[最新贡献规范](https://github.com/awesome-dsh-plugin/awesome-dsh-plugin/blob/main/contributing.md)和 PR 模板。
收录需要维护者审核；准备材料或发布自己的索引不等于已经被社区广场收录。

## 已适配的内容

| 内容 | 位置 |
|---|---|
| 单文件投稿、中英简介、workflow 分类 | [YAML 条目](shinerain1__dsh-ml-workbench.yml) |
| 预构建安装包 | 固定到 v0.4.0 GitHub Release，文件名与版本一致 |
| 市场卡片图标 | [180×180 PNG](../../assets/icon-180.png)，由已有 SVG 原样渲染 |
| 用户安装说明、依赖与范围 | [MARKETPLACE.md](../../docs/MARKETPLACE.md) |
| 声明、实现与实验的核对入口 | [EVIDENCE.md](../../docs/EVIDENCE.md) |
| 完整真实数据结果与复跑方法 | [实验报告](../../docs/REAL-DATA-EXPERIMENTS.md) |
| 维护者审阅用 PR 正文 | [PR_BODY.md](PR_BODY.md) |
| 本地格式与发布包核验记录 | [validation.json](validation.json)，记录校验器版本、条目/图标/安装包哈希 |

条目不包含 `npm`、`homepage`、工具数量等目录格式不接受的额外字段。
安装包无需用户在本机编译 TypeScript；Python 后端仍须按说明单独安装。
本次没有新增 npm 发布，也没有更改已发布的 v0.4.0 安装包。
