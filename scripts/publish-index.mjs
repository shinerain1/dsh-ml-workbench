/** Generate a format-1 DSH index pinned to a previously published source commit. */
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { resolve, join } from 'node:path'

const [commit] = process.argv.slice(2)
if (!/^[a-f0-9]{40}$/.test(commit ?? '')) throw new Error('Supply the released 40-character Git commit SHA')
const root = resolve(import.meta.dirname, '..')
const pkg = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
const icon = (await readFile(join(root, 'icon.svg'))).toString('base64')
const index = {
  format: 1,
  title: { en: 'DSH ML Workbench', zh: 'DSH 机器学习实验工作台' },
  description: { en: 'Community MLflow experiment workflow adapter', zh: '基于 MLflow 的社区实验流程适配插件' },
  plugins: [{
    name: pkg.name,
    title: { en: 'ML Experiment Workbench', zh: '机器学习实验工作台' },
    description: { en: 'Track, branch, compare and reproduce local CSV experiments with MLflow. Python backend required.',
      zh: '基于 MLflow 创建、追踪、分支、比较和复现本地 CSV 实验。需另装 Python 后端。' },
    author: 'shinerain1',
    icon: `data:image/svg+xml;base64,${icon}`,
    homepage: 'https://github.com/shinerain1/dsh-ml-workbench#readme',
    repository: { url: 'https://github.com/shinerain1/dsh-ml-workbench' },
    metadata: { keywords: ['MLflow', 'CSV', 'classification', 'regression', 'experiments', '机器学习'] },
    versions: [{ version: pkg.version, publishTimestamp: Math.floor(Date.now() / 1000),
      engines: { dsh: pkg.engines.dsh },
      source: { type: 'git', url: 'https://github.com/shinerain1/dsh-ml-workbench', commit } }],
  }],
}
await mkdir(join(root, 'market'), { recursive: true })
await writeFile(join(root, 'market', 'index.json'), JSON.stringify(index, null, 2) + '\n')
console.log(JSON.stringify({ name: pkg.name, version: pkg.version, commit, index: 'market/index.json' }))
