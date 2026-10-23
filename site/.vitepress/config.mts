import { defineConfig } from 'vitepress';
import { readFileSync } from 'node:fs';

const release = JSON.parse(readFileSync(new URL('../generated/release.json', import.meta.url), 'utf8'));
export default defineConfig({
  lang: 'zh-CN',
  title: 'Swift 实用开发笔记',
  description: '从 SwiftUI 实用开发到 CNN、生物视觉与 AI 技术的来路，沿着具体问题理解实现和原理。',
  cleanUrls: true,
  sitemap: { hostname: 'https://endlessjoker.github.io' },
  head: [['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }]],
  themeConfig: {
    logo: '/favicon.svg',
    siteTitle: 'Swift 实用开发笔记',
    nav: [
      { text: '开始阅读', link: '/reading' },
      { text: 'AI 技术的来路', link: '/ai-roots' },
      { text: '更新计划', link: '/roadmap' },
      { text: '关于', link: '/about' }
    ],
    sidebar: [
      { text: '阅读指南', items: [{ text: '从这里开始', link: '/reading' }, { text: '章节与更新计划', link: '/roadmap' }] },
      { text: 'SwiftUI 实用开发', items: release.articles.filter((a: any) => a.number <= 18).map((a: any) => ({ text: `${a.number}. ${a.title}`, link: `/articles/${a.slug}` })) },
      ...(release.articles.some((a: any) => a.number > 18 && a.number <= 31) ? [{ text: 'CNN 与生物视觉', items: release.articles.filter((a: any) => a.number > 18 && a.number <= 31).map((a: any) => ({ text: `${a.number - 18}. ${a.title}`, link: `/articles/${a.slug}` })) }] : []),
      { text: 'AI 技术的来路', items: [{ text: '系列目录', link: '/ai-roots' }, ...release.articles.filter((a: any) => a.number > 31).map((a: any) => ({ text: `${a.number - 31}. ${a.title}`, link: `/articles/${a.slug}` }))] }
    ],
    outline: { label: '本页内容', level: [2, 3] },
    docFooter: { prev: '上一篇', next: '下一篇' },
    darkModeSwitchLabel: '切换外观',
    sidebarMenuLabel: '章节目录',
    returnToTopLabel: '返回顶部',
    search: { provider: 'local', options: { locales: { root: { translations: { button: { buttonText: '搜索', buttonAriaLabel: '搜索文章' }, modal: { noResultsText: '没有找到相关内容', resetButtonTitle: '清除', footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' } } } } } } },
    socialLinks: [{ icon: 'github', link: 'https://github.com/endlessjoker/endlessjoker.github.io' }],
    footer: { message: '写一个阅读器，也记下沿途遇到的问题。', copyright: '© endlessjoker' }
  }
});
