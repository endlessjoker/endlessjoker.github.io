# Swift 实用开发笔记

Swift / SwiftUI 开发笔记，以作者自己的 epubBooks 阅读器为主线，记录需求、实现和学习过程。必要时用小例子解释代码。

网站：https://endlessjoker.github.io

## 本地预览

安装 Node.js 24 与 pnpm 11.25，然后运行：

```bash
pnpm install --frozen-lockfile
pnpm check
pnpm dev
```

生产构建与预览：`pnpm build`、`pnpm preview`。

## 发布一篇文章

1. 在 `content/` 写文章，以已获作者授权的个人 epubBooks 项目或独立小例子为素材。公司内部项目和其他未授权的材料不得引用。补充代码可放在 `examples/`，不要求每篇文章附带实验。
2. 准备公开时，在 `publication.json` 增加记录，日期使用 `YYYY-MM-DDT21:00:00+08:00`，每天最多一篇，初始 `reviewedHash` 为空。在 `.gitignore` 中为这篇文章添加例外；本地未公开的草稿保持忽略，不进入 Git 历史。
3. 核对代码、解释和来源，区分实际实现、假设的错误写法与真实经历。若文章包含配套代码，将文件登记在该文章的 `files` 列表中。
4. 完成与改动相称的代码检查、内容审阅及 `pnpm check && pnpm build`，再执行 `pnpm review <slug> --verified`，记录内容哈希。检查过程不写成读者的验收任务。
5. 运行 `pnpm check && pnpm build`，预览后提交到本仓库并推送 `main`。

GitHub Actions 每天北京时间 21:00 构建（UTC 13:00）。只有通过审阅、内容哈希未变化、且到达发布日期的文章进入网站和搜索。重复运行不提前发布。构建失败不会覆盖旧网站。

GitHub 的定时任务可能延迟；可在 Actions → Publish daily chapter → Run workflow 手动补跑。公共仓库长期没有活动时，GitHub 可能停用定时任务，需要重新启用。没有下一篇备稿时，网站维持原内容。

已经提交到 Git 仓库的待上线稿件公开可读；发布日期只控制网站连载。尚未准备公开的草稿留在本地。

## 首次启用 Pages

在仓库 Settings → Pages → Build and deployment 中将 Source 设为 GitHub Actions。工作流只请求读取内容及部署 Pages 所需的权限，不需要个人访问令牌存入仓库。

`site/public/release.json` 由构建生成，记录最近构建时间及上线文章，供发布检查使用。

技术更正应修改原文、重跑实验、更新审阅哈希并推送；不自动新增当天的章节。
