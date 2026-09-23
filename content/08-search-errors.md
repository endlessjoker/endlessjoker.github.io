# 零条结果，也可能是一本都没读到

上一篇把搜索状态理顺了，终于能分清“没搜过”和“搜完了，没有结果”。

但还有一种情况藏在后面：文件根本没读进去。

现在阅读器扫描每个文件之前，会经过这段代码：

```swift
guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
      size <= 2_000_000,
      let text = try? String(contentsOf: url, encoding: .utf8) else {
    continue
}
```

文件超过大小限制，就跳过。大小没取到，也跳过。正文读不了，还是跳过。

扫完以后只交回结果数组。于是一本文件都读不到，和所有文件都读完但没有匹配，最后都可能留下一个空数组。

界面很平静。它确实不知道发生过什么。

## 问号省下的代码，顺便省掉了原因

`String(contentsOf:encoding:)` 是可能抛错的初始化方法。文件被移动了、没有读取权限，或者内容无法按指定编码解码，都可能让读取失败。

`try?` 会在抛错时把结果变成 `nil`，原来的错误就丢掉了。外面的 `guard let` 只看有没有拿到字符串，不知道为什么没拿到。

空文件倒不属于这一种：成功读到的空字符串是 `""`，照样能通过 `guard let`。`nil` 和空内容的区别，在这里很实际。

文件大小还多一层。`fileSize` 自己就是可选值，没提供大小时也可能是 `nil`。原来这一行把“查询文件信息抛错了”和“查询返回了，但没有大小”也合在了一起。

如果只想要“能读就用，不能读就换个无关紧要的默认值”，`try?` 挺合适。可这里的失败会影响搜索范围，再把原因丢掉，我连该改关键词还是检查文件都分不清。

换成 `try!` 就更不合适了。那相当于认定这里不会抛错，真抛了会触发运行时错误。搜索不到一个词，总不能把阅读器也搭进去。

## catch 放在哪里，会决定放弃多少东西

最容易想到的改法，是把整个文件循环塞进一个 `do-catch`。

但那样第三个文件一抛错，执行就会跳出循环，进入外面的 `catch`。处理完错误，不会自动回到第四个文件接着干。

`try` 也没有“失败后再试一次”的意思。它标出这里可能抛错；一旦真的抛出，后面的正常执行路径就中断了。

我想要的行为是：这份文件没读成，记下来，继续下一份。已经找到的内容仍然有用。

所以 `do-catch` 应该放在每轮文件处理里面。它负责这一份文件的读取问题，外层循环还掌握着剩下的文件。

同样是捕获错误，位置差一层，搜索范围就可能差半个书架。

## 数组之外，还得带点消息回来

沿着上一版的改法，我准备让扫描交回这份报告：

```swift
enum SearchIssue: Sendable {
    case tooLarge(URL)
    case sizeUnavailable(URL)
    case failed(URL, message: String)
}

struct SearchReport: Sendable {
    var results: [BookSearchResult] = []
    var issues: [SearchIssue] = []
    var reachedLimit = false
}
```

`tooLarge` 是我自己定的搜索限制，不能说成文件损坏。`sizeUnavailable` 说明没拿到决定是否读取所需的大小。`failed` 才保存抛错时的说明，包括文件信息或正文读取出错。

每条问题还带着 URL，展开后能定位到具体文件。只留一个“部分文件失败”的计数，等我真想处理时，又得重新找。

`reachedLimit` 则记着另一件事：收集到 300 条时主动停止。它和文件能不能读取没有直接关系，单独放着。

这里的数组会由 worker 在局部逐步填充，最后整份交回模型。没有让后台任务直接往 SwiftUI 正在显示的列表里追加。

## 把一份文件的失败留在这一轮

下面是准备替换的 worker。匹配行和截取摘要仍然沿用原来的方式，主要把挤在一起的读取条件拆开：

```swift
let worker = Task.detached(priority: .userInitiated) { () -> SearchReport in
    var report = SearchReport()

    for (id, url) in sources {
        if Task.isCancelled { return report }

        let text: String
        do {
            let values = try url.resourceValues(forKeys: [.fileSizeKey])
            guard let size = values.fileSize else {
                report.issues.append(.sizeUnavailable(url))
                continue
            }
            guard size <= 2_000_000 else {
                report.issues.append(.tooLarge(url))
                continue
            }
            text = try String(contentsOf: url, encoding: .utf8)
        } catch {
            report.issues.append(.failed(url, message: error.localizedDescription))
            continue
        }

        for (line, value) in text.components(separatedBy: .newlines).enumerated() {
            if Task.isCancelled { return report }
            guard let range = value.range(
                of: query, options: [.caseInsensitive, .diacriticInsensitive]
            ) else { continue }

            let start = value.index(range.lowerBound, offsetBy: -45,
                                    limitedBy: value.startIndex) ?? value.startIndex
            let end = value.index(range.upperBound, offsetBy: 100,
                                  limitedBy: value.endIndex) ?? value.endIndex
            report.results.append(BookSearchResult(
                library: id, url: url, line: line + 1,
                snippet: String(value[start..<end])
            ))

            if report.results.count >= 300 {
                report.reachedLimit = true
                return report
            }
        }
    }
    return report
}
```

`let text: String` 先声明，等读取成功才赋值。拿不到大小、文件过大、抛出错误，这几条路都直接 `continue`，不会走到后面的正文匹配。能走到匹配的路径，就一定已经给 `text` 赋过值。

`catch` 中的 `error` 是这次捕获的错误。这里先留下系统提供的文字说明，供界面展开查看。如果以后要按错误种类决定重新授权或重试，还需要保留错误本身，或相应的错误域与代码；不能拿一段会随语言变化的 `localizedDescription` 当稳定标识。

捕获以后选择继续，是这个搜索功能的决定。Swift 没有规定“用了 catch 就该忽略”，也没有规定“遇到 throw 就必须整件事作废”。如果把读取提成一个 `throws` 方法，它也只是把处理责任交给调用处，最后仍然要有人决定放弃到哪一层。

代码里的大小判断也只是读取前的检查。文件仍然可能在检查之后变大或者被替换，它不是对整个读取过程的硬性内存限制。

## 取消搜索，不能算文件坏了

上面的 `catch` 只围住查询大小和同步读取。取消仍然用 `Task.isCancelled` 检查，直接返回当前累积的报告，不往 `issues` 里塞一个“读取失败”。

这次还在逐行匹配时加了检查，避免已经取消却继续匹配整个文件。不过正在进行的同步读取，以及把正文拆成行的操作，还是不会被这些检查强行中断。

提前返回的报告可能不完整。它会回到上一层，继续经过原来的取消转发和回写检查，只是返回值换成了报告：

```swift
let report = await withTaskCancellationHandler {
    await worker.value
} onCancel: {
    worker.cancel()
}

guard !Task.isCancelled,
      searchState.pendingRequestID == request.id else { return }

searchState = .finished(SearchBatch(request: request, report: report))
```

这里沿用现有的取消关系：外层请求被取消，才会通知这个 worker 取消；外层回写前还要检查自己的取消状态。worker 交回一份报告，不代表它就有资格显示为完成。

以后如果给 worker 增加独立的取消入口，就得把“这份报告是取消后返回的”也明确带出来，不能只依赖外层任务的标记。

这也是 `catch` 不能乱扩大的原因。如果以后把 `try await Task.sleep(...)` 或 `try Task.checkCancellation()` 放进文件读取的 `do` 里，它们抛出的 `CancellationError` 也会被这个无条件的 `catch` 接住。到时候必须先区分取消，让任务退出，不能记成一条文件问题后继续扫。

我只是又敲了一个字，上一轮搜索就开始给教材报错，多少有点冤。

## 搜索完成，可以带着几条问题

上一章的 batch 也要跟着调整，把结果数组的位置换成报告：

```swift
struct SearchBatch: Sendable {
    let request: SearchRequest
    let report: SearchReport
}
```

请求仍然记着这批结果对应的查询词，列表的数据改从 `batch.report.results` 取。`.finished` 表示当前这轮可展示的处理结束了，有没有跳过文件、是否达到上限，要看报告。

例如一份文件超过限制，另外两份读到了命中内容，我想看到的是结果列表，以及可以展开的跳过说明。没必要盖上一个大大的“搜索失败”，把还能用的内容一起挡住。

有了这些数据，零条结果时也能说得更准确：

| 扫描报告 | 可以显示的意思 |
|---|---|
| 没有匹配，也没有记录问题 | 本次文件清单中没有找到匹配。 |
| 没有匹配，记录了跳过或读取问题 | 在已搜索的内容中没有匹配，有文件未参与。 |
| 有匹配，也有问题 | 保留结果，同时提供未参与文件的详情。 |
| 达到 300 条上限 | 已列出 300 条，可能还有更多。 |

最后一种可以和前面的文件问题同时出现。即使第 300 条恰好就是最后一个文件的最后一个匹配，这段代码也没有继续检查，因而只能说“可能还有更多”。

而“没有记录问题”也只针对这次拿到的 `sources`。目录扫描时就没进入清单的文件，不会凭空出现在这份报告里。

## Result 也得装得下这些信息

只把返回类型改成 `Result<[BookSearchResult], Error>`，还没决定部分成功怎么办。

它的值要么是 `.success`，要么是 `.failure`。如果一个文件出错就返回失败，前面找到的结果放哪里？如果返回成功数组，跳过的原因又丢到哪里？

当然可以让成功值装 `SearchReport`。问题从来不是 `Result` 放不下，而是我需要先承认：这次搜索可以拿到一些可用结果，同时带着几条文件问题。

目前这些文件级错误都在循环里处理了，worker 可以正常返回报告，不必为了用了 `try`，就给整个扫描接口加一个 `throws`。将来真的存在“整轮都没法继续”的失败，再决定怎样往上传。

这一版还只是改法，阅读器当前仍会静默跳过读取失败的文件。接下来值得继续拆的是模型与磁盘之间的边界：如果每次想看看失败提示，都得真的搬走一本教材、改一次权限，写这个小阅读器也未免太费书了。
