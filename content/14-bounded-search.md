# 一次搜很多本书，也不能把任务开满

全文搜索现在是一份后台任务，拿着文件清单逐本往下读。每本最多 2 MB，找到 300 条就停。它有个好处：顺序很好懂。结果跟着文件清单走，不会因为哪本书先读完，就突然插到前面。

如果教材文件越来越多，我可能会想让几本书同时搜索。最顺手的写法，是把循环换成任务组，每本书加一个子任务。写着写着会发现，文件清单有多少项，就能一口气加多少个任务。任务组负责收齐孩子，不负责替我猜“这个磁盘同时读几本比较合适”。

当前阅读器还没有并行搜索。下面是沿着第八篇的 `SearchReport` 继续设计的改法，不是一次已经测出收益的优化。

## 先想清楚要保留什么

旧搜索遇到读不进来的文件会静默跳过。第八篇已经准备把单文件的匹配结果和读取问题一起放进报告。并行以后，这个决定更不能丢：一本文件失败，不该让另外三本已经找到的内容作废。

另外还有两条约束。最多只让四份文件扫描同时进行；结果继续按原文件清单排列。四不是测出来的“最佳线程数”，只是先给同时进行的读取设一个清楚的上限，实际合适的值还得量。

如果只写下面这种循环，就没有上限：

```swift
await withTaskGroup(of: FileReport.self) { group in
    for (index, source) in sources.enumerated() {
        group.addTask {
            scanOne(source, index: index, query: query, limit: 300)
        }
    }
}
```

这个片段只是说明“全加进去”的写法，连结果都没收集。`addTask` 创建的是任务组的子任务，不是排队等前一个完成才创建下一个。任务组会在离开作用域前等孩子结束，但这不等于它自动限制同时开始的文件读取。

## 四本一批，先把上限写死

这一版不用急着追求最复杂的调度。每批最多四本，等这一批全部交回报告，再决定要不要开下一批：

```swift
struct FileReport: Sendable {
    let index: Int
    let results: [BookSearchResult]
    let issues: [SearchIssue]
}

nonisolated func scanFilesInBatches(
    sources: [(UUID, URL)], query: String
) async -> SearchReport {
    var report = SearchReport()
    let width = 4

    for start in stride(from: 0, to: sources.count, by: width) {
        if Task.isCancelled { return report }
        let end = min(start + width, sources.count)

        let batch = await withTaskGroup(
            of: FileReport.self, returning: [FileReport].self
        ) { group in
            for index in start..<end {
                let source = sources[index]
                group.addTask {
                    scanOne(source, index: index, query: query, limit: 300)
                }
            }

            var parts: [FileReport] = []
            for await part in group { parts.append(part) }
            return parts.sorted { $0.index < $1.index }
        }

        if Task.isCancelled { return report }
        for part in batch {
            report.issues.append(contentsOf: part.issues)
            let room = 300 - report.results.count
            report.results.append(contentsOf: part.results.prefix(room))
        }
        if report.results.count >= 300 {
            report.reachedLimit = true
            return report
        }
    }
    return report
}
```

这里的 `scanOne` 是把第八篇“读取一份文件、逐行匹配、记录问题”的代码提成同步函数。它最多保留 300 条单文件命中，返回自己的 `index`、命中和问题；取消时检查 `Task.isCancelled`，不要把取消记成“文件损坏”。每个子任务只写自己的 `FileReport`，最后由外层合并，不让四个任务同时修改同一份结果数组。

`for await part in group` 收到的是完成顺序。一本短文件可能先回来，哪怕它排在书单末尾。所以我给每份报告带原来的 `index`，一批全部回来后再排序。否则“加了并发”会顺手把列表排序也改掉，功能表面上没报错，看起来却像书架自己洗了牌。

合并时先收这一批所有已处理文件的问题，再按原顺序取最多 300 条结果。达到 300 条以后不启动下一批。本批里排在后面的文件可能也已经读完，只是多余的命中没有进入展示数组。因此 `reachedLimit` 仍表示“到达展示上限，可能还有更多”，不能把它说成“所有教材都搜完了”。

这个批次方案比较保守：四本里有一本特别慢，其余三本已经结束，也要等慢的回来才开下一批。好处是结果顺序、上限和部分问题都比较容易说清楚。以后真的看到这里成为瓶颈，再改成“完成一个就补一个”的滑动窗口；那时还得处理提前达到 300 条时，已经在跑的后续文件怎样收尾。

## 错误留在文件里，取消交给整轮

这里选普通的 `withTaskGroup`，因为单文件读取错误会转成 `SearchIssue`，放在 `FileReport` 里。某个子任务不需要抛出错误来终止整个组。一本书读不成，另几本书找到的内容依然有用。

这不意味着把所有错误都吞掉。文件大小取不到、正文解码失败、超过自定大小限制，仍要分清原因，交给第八篇的报告结构。单文件函数要保证出了问题也能正常返回一份带 `issues` 的报告。

取消是另一回事。任务组是结构化的：外面的搜索任务被取消，组里的子任务会收到取消信号。它们正在执行 `String(contentsOf:)` 这样的同步读取时，不会被瞬间拔掉；要等到能检查取消的位置，才会尽快退出。`withTaskGroup` 也会等已经创建的子任务结束，再从作用域返回。

所以函数里取消时返回的那份 `report` 可能只装了一部分。沿用前面几篇的回写检查，外层搜索模型看到旧请求已取消，就不能把这份半成品标成“搜索完成”。任务组负责孩子的生命周期，模型仍负责判断这轮结果还有没有资格显示。

## 限制任务数，也不等于限制所有内存

一次只开四份文件读取，比把整个清单同时扔进去更可控。但每份文件还会读成字符串、拆成行、产生摘要；前面的 2 MB 文件大小检查也不是读取过程的硬性内存封顶。若文件在检查后变化，或者一次匹配留下很多字符串，内存占用仍可能高于“4 × 2 MB”这种口算。

这篇的上限是**同时扫描的文件数**，不是磁盘吞吐、线程数或内存用量的实测保证。四这个数字也不是越大越好：读同一块盘，任务多了不一定更快，反倒可能把取消后的收尾拖长。

现在的顺序循环未必需要立刻替换。它简单，结果顺序稳定，而且目前没有测量能证明它慢到值得加这层调度。把任务组写出来，是为了以后真要并行时知道自己在换什么：除了速度，还会碰到上限、排序、部分失败和取消。少想任何一个，搜索框倒是热闹了，结果未必更可信。
