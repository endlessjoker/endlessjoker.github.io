# 缓存一直留着，内存迟早不答应

阅读器为了显示“这篇文件和上次读到的有什么变化”，会留一份上次的正文。第一次看到 `lastReadText:` 这样的键，我有点想把所有读过的内容都留着：反正下次可能还会翻回来，省得再读一遍。

但“下次可能用到”这句话特别能吃空间。给每一篇都找一个理由，最后就是谁也舍不得扔。

## 现有快照已经有两道栏杆

当前 `recordUpdate` 没有无限保存正文。新读到的文本不超过 1,000,000 个 UTF-8 字节时，它才把文本写进 `UserDefaults`；`snapshotPaths` 记录路径，重新读到同一篇就把路径挪到末尾，超过 20 篇则从最前面删除。

这两道限制管的是不同东西。只限 20 篇，可能留下 20 篇都接近大小上限的内容；只限单篇大小，则篇数仍可不断增加。两条一起用，才给这个“上次读到的正文”留了边界。

它也不是本文准备添加的**内存正文缓存**。`UserDefaults` 是持久设置存储；这里保存快照，是为了下次还能比较文件变化。限制了写入的文本量，并不等于精确限制了进程内存。编码、容器、系统自己的缓存，还有当前正在显示的正文，都不算在“20 × 1 MB”这笔口算里。

模型里另有 `libraryChapters: [UUID: [Chapter]]`，保存已添加阅读位置的文件清单。扫描会更新它，移除阅读位置时会删掉对应条目。它确实占内存，但也正在给侧栏提供目录；如果为了省空间直接把一项淘汰，侧栏还没安排重扫，就可能突然没书可列。不是凡是存在字典里的数据，都能当作“删了也无所谓”的缓存。

## 如果想缓存正文，先说清能不能扔

假设以后切换章节时反复读取同几篇，测量之后发现读盘值得省，我才会加一层临时正文缓存。它的契约应该是：命中时可以少读一次；没命中就重新读取；清空它不能让阅读器丢失文档、授权或用户笔记。当前项目还没有这层多文件正文缓存。

下面这份小缓存按 URL 找正文，并保存读取时对应的文件修改时间：

```swift
nonisolated struct TextVersion: Hashable, Sendable {
    let url: URL
    let modifiedAt: Date
}

actor TextCache {
    private struct Entry {
        let version: TextVersion
        let text: String
        let bytes: Int
    }

    private var entries: [URL: Entry] = [:]
    private var recent: [URL] = []
    private var usedBytes = 0
    private let budget = 8_000_000

    func text(for version: TextVersion) -> String? {
        guard let entry = entries[version.url] else { return nil }
        guard entry.version == version else {
            remove(version.url)
            return nil
        }
        recent.removeAll { $0 == version.url }
        recent.append(version.url)
        return entry.text
    }

    func insert(_ text: String, for version: TextVersion) {
        remove(version.url)
        let cost = text.utf8.count
        guard cost <= budget else { return }

        entries[version.url] = Entry(version: version, text: text, bytes: cost)
        recent.append(version.url)
        usedBytes += cost

        while usedBytes > budget, let oldest = recent.first {
            remove(oldest)
        }
    }

    func remove(_ url: URL) {
        if let old = entries.removeValue(forKey: url) {
            usedBytes -= old.bytes
        }
        recent.removeAll { $0 == url }
    }
}
```

`recent` 最后面是最近被读或写的文件；要腾空间，就从最前面删。重新放入同一 URL 前先移除旧值，账本才不会把一篇算两次。单篇正文比预算还大，直接不缓存；它仍可以照常显示，只是下一次还得重新读取。缓存不是阅读权限的替身，也不该决定文件能不能打开。

这里用 `text.utf8.count` 给文本记成本，是一套**缓存自己的账**。它能约束保留在 `entries` 里的 UTF-8 字节总数，却不是进程内存硬上限。Swift 字符串的实际存储、字典和数组的开销、正在读取或渲染的临时副本，仍在账本外。八百万这个数也只是例子，没有经过当前阅读器的内存测量。

为了把算法写直，`recent.removeAll` 每次会扫一遍顺序数组。预算小、条目少时容易看懂；真的缓存了成千上万篇，再换链表或别的索引，而不是先造一套复杂结构庆祝还没出现的性能问题。

## 腾出空间，跟内容还新不新是两回事

最旧的缓存条目被删，是**淘汰**。文件内容变了，旧文本不该再命中，是**失效**。这两个词很像，处理原因却不同。

`text(for:)` 会核对 `modifiedAt`，不一致就删旧条目并返回未命中。阅读器现有的文件监看也已经用修改时间判断是否要重载当前文档；把这份信息交给缓存，至少不会只看路径就一直端出旧正文。

但修改时间不是魔法封印。读元数据和读正文之间，文件可能被改动；某些替换方式也可能让时间戳不足以区别内容。如果真要把缓存用在读盘路径上，读取前后得核对文件版本，明确重载时还应主动失效。查不到可靠的修改时间，就别硬把旧内容当命中。缓存错一次，比多读一次更烦。

阅读位置被移除时，还要删掉它对应的缓存条目。尤其这里的文件可能来自安全作用域书签：失去访问授权以后，不该因为内存里还留着旧文本，就绕过原本的访问判断继续展示。缓存查询必须发生在“当前仍允许打开这份文件”的判断之后；这一点不能交给 `TextCache` 根据 URL 自己猜。

如果用系统的 `NSCache`，它能在压力下自动丢弃内容，也能设置 `totalCostLimit`。不过那个限制不是严格的字节预算，淘汰顺序也不保证是最近最少使用。需要“约束这份字典里累计的文本成本”时，我会像上面这样自己记账；如果只想要可随时丢弃的机会性缓存，`NSCache` 反而省事。两种都不能代替文件版本检查和授权判断。

现在阅读器真正需要的，可能还只是那份有上限的变化快照，以及当前文件的 `markdown`。多文件正文缓存得先证明它值得存在。缓存一旦加上，代码要同时回答三个问题：最多留多少、什么时候变旧、先扔谁。只回答“读过就存”，内存迟早替我回答。
