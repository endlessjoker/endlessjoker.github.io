import { readFileSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
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
for (const dir of ['articles', 'labs', 'generated', 'public/examples']) {
  rmSync(path.join(site, dir), { recursive: true, force: true });
  mkdirSync(path.join(site, dir), { recursive: true });
}
const release = { generatedAt: now.toISOString(), timezone: 'Asia/Shanghai', articles: articles.map(({ number, slug, title, publishAt }) => ({ number, slug, title, publishAt })) };
for (const file of ['generated/release.json', 'public/release.json']) writeFileSync(path.join(site, file), JSON.stringify(release, null, 2) + '\n');

for (const [index, article] of articles.entries()) {
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
  tagline: 从一个小小的阅读器写起，聊聊 SwiftUI、异步代码，还有那些看起来没毛病的写法。
  actions:
    - theme: brand
      text: ${articles.length ? '开始阅读第一篇' : '了解阅读方法'}
      link: ${articles.length ? `/articles/${articles[0].slug}` : '/reading'}
    - theme: alt
      text: 查看更新计划
      link: /roadmap
features:
  - title: 每篇解决一个问题
    details: 从打开文件、切换章节、处理异步结果这些小功能出发，把遇到的问题慢慢讲清楚。
  - title: 跟着代码想一想
    details: 先看直觉上的写法，再顺着出错的地方，弄明白究竟发生了什么。
  - title: 按顺序逐步学习
    details: 北京时间每天 21:00 发布已检查的备稿，没写完的就再等等。
---

## 最新更新

${latest ? `**第 ${latest.number} 篇 · [${latest.title}](/articles/${latest.slug})**\n\n${latest.summary}\n\n${latest.publishAt.slice(0, 10)} 发布 · 约 ${latest.minutes} 分钟` : '第一篇正在准备。可以先阅读[学习方法](/reading)。'}

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
writeFileSync(path.join(site, 'roadmap.md'), `# 章节与更新计划\n\n这一轮计划 ${series.totalChapters} 篇，目前 ${plan.articles.filter(item => item.reviewedHash).length} 篇已检查并排期，后续还有 ${future.length} 篇尚未排期。\n\n北京时间每天 21:00 发布一篇已检查的备稿。GitHub 定时服务可能延迟；没有合格备稿时保持已有内容。\n\n| 篇目 | 主题 | 北京时间 | 状态 |\n|---|---|---|---|\n${rows.join('\n')}\n\n## 后续写作安排\n\n| 篇目 | 暂定标题 | 主要内容 |\n|---|---|---|\n${futureRows.join('\n')}\n\n后续主题会随阅读器的实际开发调整，完成写作和检查后再安排上线日期。\n`);

console.log(`Prepared ${articles.length} published articles; next: ${plan.articles[articles.length]?.publishAt ?? 'none'}.`);
