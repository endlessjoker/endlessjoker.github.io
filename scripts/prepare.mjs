import { readFileSync, writeFileSync, mkdirSync, rmSync, copyFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { releasedArticles, verifyReviewedFiles } from './publication.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const plan = JSON.parse(readFileSync(path.join(root, 'publication.json')));
verifyReviewedFiles(plan, root);
// Production always uses the real clock. A manual workflow rerun cannot unlock tomorrow.
const now = new Date();
const articles = releasedArticles(plan, now);
const site = path.join(root, 'site');
const assetRoots = [
  ['content/assets/', 'public/cnn-vision/'],
  ['content/transformer-data/', 'public/transformer-data/']
];
for (const dir of ['articles', 'labs', 'generated', 'public/examples', ...assetRoots.map(([, local]) => local)]) {
  rmSync(path.join(site, dir), { recursive: true, force: true });
  mkdirSync(path.join(site, dir), { recursive: true });
}
const release = { generatedAt: now.toISOString(), timezone: 'Asia/Shanghai', articles: articles.map(({ number, slug, title, publishAt }) => ({ number, slug, title, publishAt })) };
for (const file of ['generated/release.json', 'public/release.json']) writeFileSync(path.join(site, file), JSON.stringify(release, null, 2) + '\n');

for (const [index, article] of articles.entries()) {
  for (const [sourceRoot, publicRoot] of assetRoots) {
    for (const file of article.files.filter(file => file.startsWith(sourceRoot))) {
      const target = path.join(site, publicRoot, file.slice(sourceRoot.length));
      mkdirSync(path.dirname(target), { recursive: true });
      copyFileSync(path.join(root, file), target);
    }
  }
  const frontmatter = {
    title: article.title, description: article.summary,
    prev: index ? { text: articles[index - 1].title, link: `/articles/${articles[index - 1].slug}` } : { text: '从这里开始', link: '/reading' },
    next: index + 1 < articles.length ? { text: articles[index + 1].title, link: `/articles/${articles[index + 1].slug}` } : { text: '后续更新计划', link: '/roadmap' }
  };
  const source = readFileSync(path.join(root, article.article), 'utf8');
  const date = article.publishAt.slice(0, 10);
  const body = source.replace(/^(# .+)$/m, `$1\n\n<div class="chapter-meta">第 ${article.number} 篇 · ${date} · ${article.minutes} 分钟阅读</div>`);
  writeFileSync(path.join(site, 'articles', `${article.slug}.md`), `---\n${JSON.stringify(frontmatter)}\n---\n\n${body}`);
}

const latest = articles.at(-1);
const index = `---
layout: home
hero:
  name: Swift 实用开发笔记
  text: 从一个能复现的问题开始。
  tagline: 从 SwiftUI 到 PyTorch 和 Transformer，用具体计算和代码理解原理。
  actions:
    - theme: brand
      text: ${articles.length ? '开始阅读第一篇' : '了解阅读方法'}
      link: ${articles.length ? `/articles/${articles[0].slug}` : '/reading'}
    - theme: alt
      text: 查看更新计划
      link: /roadmap
features:
  - title: 每篇解决一个问题
    details: 从具体问题出发，解释计算方法背后的取舍。
  - title: 跟着实验想一想
    details: 结合代码、交互实验与原始文献，检查直觉成立的条件。
  - title: 按顺序逐步学习
    details: 北京时间每天 21:00 发布已检查的备稿，没写完的就再等等。
---

## 最新更新

${latest ? `**第 ${latest.number} 篇 · [${latest.title}](/articles/${latest.slug})**\n\n${latest.summary}\n\n${latest.publishAt.slice(0, 10)} 发布 · 约 ${latest.minutes} 分钟` : '第一篇正在准备。可以先阅读[学习方法](/reading)。'}

## 专题阅读

[Transformer 与开源大模型](/transformer)——用实际检查点复算注意力、缓存、专家路由和解码。

[看得懂的 PyTorch：从理论到代码](/pytorch)——从两个参数开始，逐步算清模型怎样学习。

[AI 技术的来路：18 讲目录](/ai-roots)——从 RNN 的记忆开始，追到注意力、Transformer 与大模型。

## 已发布文章

${articles.map(a => `- [${a.title}](/articles/${a.slug}) — ${a.summary}`).join('\n')}
`;
writeFileSync(path.join(site, 'index.md'), index);

const rows = plan.articles.map(item => {
  const live = articles.some(a => a.slug === item.slug);
  const title = live ? `[${item.title}](/articles/${item.slug})` : item.title;
  return `| ${item.number} | ${title} | ${item.publishAt.slice(0, 10)} ${item.publishAt.slice(11, 16)} | ${live ? '已发布' : item.reviewedHash ? '已检查，待上线' : '准备中'} |`;
});
const series = JSON.parse(readFileSync(path.join(root, 'series-plan.json')));
const future = series.chapters.filter(topic => !plan.articles.some(item => item.number === topic.number));
const futureRows = future.map(topic => `| ${topic.number} | ${topic.title} | ${topic.focus} |`);
writeFileSync(path.join(site, 'roadmap.md'), `# 章节与更新计划\n\nSwiftUI、CNN 与生物视觉、AI 技术的来路、看得懂的 PyTorch 和 [Transformer 与开源大模型](/transformer)共规划 ${series.totalChapters} 篇，其中 ${plan.articles.filter(item => item.reviewedHash).length} 篇已检查并排期。\n\nPyTorch 第十八讲排在 2026 年 11 月 30 日，Transformer 系列从 12 月 1 日接续。每天北京时间 21:00 发布一篇已检查的备稿。GitHub 定时服务可能延迟；没有合格备稿时保持已有内容。\n\n| 篇目 | 主题 | 北京时间 | 状态 |\n|---|---|---|---|\n${rows.join('\n')}\n\n## 后续写作安排\n\n${futureRows.length ? '| 篇目 | 暂定标题 | 主要内容 |\n|---|---|---|\n' + futureRows.join('\n') : '表中的日期以外暂无排期。'}\n`);

console.log(`Prepared ${articles.length} published articles; next: ${plan.articles[articles.length]?.publishAt ?? 'none'}.`);

const aiRows = series.chapters.filter(topic => topic.number > 31 && topic.number <= 49).map(topic => {
  const live = articles.find(a => a.number === topic.number);
  const scheduled = plan.articles.find(a => a.number === topic.number);
  const title = live ? `[${topic.title}](/articles/${live.slug})` : topic.title;
  const status = live ? '已发布' : scheduled?.reviewedHash ? '已检查，待上线' : '写作计划';
  return `| ${topic.number - 31} | ${title} | ${status} |`;
});
writeFileSync(path.join(site, 'ai-roots.md'), `# AI 技术的来路

模型名称越来越多，真正需要弄清的是：前一个办法卡在了哪里，新的办法改了哪一步。

这个系列计划 18 讲，从 RNN、LSTM 和序列翻译开始，逐步讲到 QKV、Transformer 与后续模型。每讲围绕一个问题展开，用小算例解释计算，把原始论文名称直接放在引用处。

## 阅读顺序

| 讲次 | 主题 | 状态 |
|---|---|---|
${aiRows.join('\n')}

已有基础可以按问题选读；第一次接触注意力，建议按前五讲的顺序。写作计划中的文章完成后再排期，具体上线时间见[更新计划](/roadmap)。

## 关于材料

本系列根据研究论文独立编写，教学算例另行标明，不是某本书的逐章译文。讨论历史模型时，会保留当时的实验条件，避免把后来的能力倒算到前人的工作上。
`);

const pytorchRows = series.chapters.filter(topic => topic.number >= 50 && topic.number <= 67).map(topic => {
  const live = articles.find(a => a.number === topic.number);
  const scheduled = plan.articles.find(a => a.number === topic.number);
  const title = live ? `[${topic.title}](/articles/${live.slug})` : topic.title;
  const status = live ? '已发布' : scheduled?.reviewedHash ? '已检查，待上线' : '写作计划';
  return `| ${topic.number - 49} | ${title} | ${status} |`;
});
writeFileSync(path.join(site, 'pytorch.md'), `# 看得懂的 PyTorch

从四条数据、两个参数开始，把模型的预测、损失、梯度和更新落实到代码。每讲使用可手算的例子，代码可按正文顺序运行，普通 CPU 即可。

## 阅读顺序

| 讲次 | 主题 | 状态 |
|---|---|---|
${pytorchRows.join('\n')}

全系列 18 讲已完成并核对，覆盖训练基础、批量训练、分类、标准化与保存部署。实验使用 PyTorch 2.13.0，具体上线时间见[更新计划](/roadmap)。
`);

const transformerRows = series.chapters.filter(topic => topic.focus.startsWith('Transformer 与开源大模型 · ')).map(topic => {
  const live = articles.find(a => a.number === topic.number);
  const scheduled = plan.articles.find(a => a.number === topic.number);
  const title = live ? `[${topic.title}](/articles/${live.slug})` : topic.title;
  const label = topic.focus.split(' · ')[1];
  const date = scheduled?.publishAt.slice(0, 10) ?? '待定';
  return `| ${label} | ${title} | ${date} | ${live ? '已发布' : scheduled?.reviewedHash ? '待上线' : '写作计划'} |`;
});
writeFileSync(path.join(site, 'transformer.md'), `# Transformer 与开源大模型

同一条「法国的首都是巴黎。德国的首都是」，可以追到词向量、真实的 Q、K、V，再看到不同模型怎么存过去的 token、怎么挑专家。投机解码两讲另用完整的小模型与 DeepSeek 的草稿头，拆开「先猜，再验」的计算。

正文中的数值来自固定版本的官方权重，论文核算和教学例子单独标明。带数据的篇目附了下载，PyTorch 在 CPU 上就能复算。

| 讲次 | 主题 | 上线日期（北京时间 21:00） | 状态 |
|---|---|---|---|
${transformerRows.join('\n')}

需要补齐张量与矩阵乘法，可以先读[看得懂的 PyTorch](/pytorch)。全站排期见[章节与更新计划](/roadmap)。
`);
