# 为了看个错误提示，没必要真把文件弄坏

前两篇给搜索设计了状态和报告。结果为空、文件太大、部分读取失败，终于各有地方可放。

接下来想看一眼这些提示长什么样，就遇到了一个小麻烦：目前阅读器的 `searchBooks()` 会直接创建 worker，查询文件大小，再从磁盘读正文。

如果一直沿用这条路，想看失败提示，就得准备一份真的读不了的文件；想看加载状态，就得碰上一轮足够慢的搜索。文件太小，转圈可能还没看清就没了。

我只想调个界面，总不能先给教材制造一点生活困难。

## 模型需要的，其实是一份搜索报告

当前 `ReaderModel` 同时知道输入框写了什么、正在等哪个请求、怎样读取文件，以及结果该什么时候显示。

后面两种知识可以分开。模型只需要把文件清单和查询词交出去，等一份 `SearchReport` 回来。至于报告来自真实扫描，还是预览用的固定数据，不必写进模型的分支里。

沿着前两篇的设计稿，我准备抽出一个只管搜索的小模型。文件授权、目录管理、打开章节这些事仍留在原来的位置。这还不是阅读器已经完成的重构。

先把搜索操作写成一个函数类型：

```swift
typealias SearchOperation = @MainActor (
    [(UUID, URL)], String
) async -> SearchReport
```

前一个参数是“阅读位置 ID + 文件 URL”的清单，后一个是查询词，返回值沿用上一篇的报告。

函数也可以当作值传递。模型可以保存这份操作，等有输入时再调用，不一定非得把它写死成自己的某个方法。

这里还明确写了 `@MainActor`：模型从 MainActor 调用这个入口，入口本身也在这个隔离范围里。`async` 允许它等待后续工作，不代表整个函数自动去后台执行。

## 构造模型时，把搜索方法交给它

把输入、状态和任务放进小模型，文件清单则由调用处提供：

```swift
@MainActor
@Observable
final class SearchModel {
    var query = ""
    private(set) var state: SearchState = .idle

    private let scan: SearchOperation
    private var task: Task<Void, Never>?

    init(scan: @escaping SearchOperation) {
        self.scan = scan
    }

    func update(sources: [(UUID, URL)]) {
        task?.cancel()

        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            state = .idle
            return
        }

        let request = SearchRequest(query: text)
        let previous = state.displayedBatch
        state = .waiting(request, previous: previous)

        task = Task {
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled,
                  state.pendingRequestID == request.id else { return }

            state = .searching(request, previous: previous)
            let report = await scan(sources, request.query)

            guard !Task.isCancelled,
                  state.pendingRequestID == request.id else { return }

            state = .finished(SearchBatch(request: request, report: report))
        }
    }
}
```

防抖、取消旧请求、回写前核对身份，都还是原来的安排。变化集中在中间：模型调用保存下来的 `scan`，不再亲自决定怎么读文件。

`sources` 是这次调用传进来的值。任务等待期间，外面的目录清单即使更新了，也不会让本轮扫描临时换一份输入。如果需要按新清单重新搜索，调用处要再发起一次 `update`。

初始化参数上的 `@escaping` 说的是这份闭包可以活得比这次 `init` 调用更久。模型把它存进属性，初始化结束以后还要用，所以需要这个标记。

它和“逃到后台线程”没有关系。一个 escaping 闭包完全可以以后仍在 MainActor 上调用。什么时候保存、什么时候调用、在哪个隔离范围里调用，是几件不同的事。

这就是这里的依赖注入：创建模型时，把它需要的搜索能力交给它。名字挺正式，落到这段代码里，就是一个初始化参数。

## 真正的读取，仍然得安排执行位置

真实版本的入口可以接成这样：

```swift
@MainActor
func searchOnDisk(
    sources: [(UUID, URL)], query: String
) async -> SearchReport {
    let worker = Task.detached(priority: .userInitiated) {
        scanFiles(sources: sources, query: query)
    }

    return await withTaskCancellationHandler {
        await worker.value
    } onCancel: {
        worker.cancel()
    }
}
```

这里的 `scanFiles` 是把上一篇 worker 里的循环原样提成一个同步辅助函数，返回 `SearchReport`，声明为 `nonisolated`。它只使用传入的文件清单和查询词，保留文件间、逐行的取消检查，不读取模型状态。

项目采用默认 MainActor 隔离，所以这个同步辅助函数要明确自己的隔离边界。`nonisolated` 本身不会安排后台执行；真正调用它的是上面的 detached 任务。

`searchOnDisk` 留在 MainActor，负责启动工作和等待报告。磁盘读取与匹配发生在 worker 里。跨过去的是文件清单和字符串，回来的报告也符合 `Sendable`，没有把整个模型扔给 worker。

创建正式使用的搜索模型时，传入这个函数即可：

```swift
let searchModel = SearchModel(scan: searchOnDisk)
```

这里传的是函数，不是在创建模型时就开始扫描。后面调用 `update`，经过防抖，才会用这份函数处理具体输入。

如果只是把读取代码从模型搬到另一个文件，然后在 MainActor 里同步调用，卡顿的可能性一点没少。换文件位置不会顺便换执行位置。

同样，这个拆分也不会获得新的文件访问权限。真实扫描使用的 URL 仍然依赖原来的授权和访问生命周期，不能因为现在有了一个函数参数，就忘掉前面处理书签时的那些事。

## 预览时，换一份会故意慢下来的实现

我想看的页面状态，是等一会儿以后出现一条结果，同时有一个文件读取失败。可以在构造时换成这份操作：

```swift
@MainActor
func makeSearchPreviewModel() -> SearchModel {
    SearchModel { _, query in
        try? await Task.sleep(for: .seconds(2))

        let library = UUID()
        let chapter = URL(fileURLWithPath: "/preview/chapter.md")
        let unavailable = URL(fileURLWithPath: "/preview/unavailable.md")

        return SearchReport(
            results: [BookSearchResult(
                library: library, url: chapter, line: 12,
                snippet: "包含 \(query) 的演示片段"
            )],
            issues: [.failed(unavailable, message: "演示：这个文件无法读取")],
            reachedLimit: false
        )
    }
}
```

这份实现故意忽略文件清单，只根据查询词造一份报告。两个 URL 都是演示用的标识，没有去访问这些路径，预览里的条目也不接真实打开文件的操作。

它不需要知道模型是否正显示转圈，也不直接修改 `state`。等待开始和结束时显示什么，还是同一个 `SearchModel` 负责。

要看纯空结果，就让闭包返回空报告；要看达到上限的提示，就准备相应的报告。模型里面不用出现 `if isPreview`，真实扫描也不用混进“偶尔故意失败一次”的调试分支。

这里的两秒只是为了让加载提示留得久一点，不是精确的时序保证。它再加上模型原有的防抖，才是一次输入等待的大致过程。

而且 `try?` 吞掉取消以后，这个替身仍会继续构造报告。模型回写前的检查正好不能省：即使依赖交回了数据，也要由模型判断这份请求是否还有效。

真实扫描内部的取消转发用于节省工作，模型的回写检查用于保护当前状态。把读取换成替身以后，这两件事的分工反而更清楚了。

## 现在需要 protocol 吗

目前这个边界只有一个操作：给清单和词，返回报告。一个闭包就能表达，暂时不需要为了“可替换”先配齐一串协议和实现类型。

以后如果搜索服务还要维护索引、监听目录、处理刷新和停止，这些操作共享自己的状态，做成一个有明确接口的类型就更合适了。到时候再考虑协议，也有具体的东西可抽象。

如果多份任务要访问同一份可变索引，actor 可能负责保护那份状态。但给同步磁盘扫描套上一个 actor，不会让读取变成可中断的异步 I/O；即使离开 MainActor，同步读取仍会占用执行它的线程。

现在这一步解决的是模型不必绑死一种搜索实现。线程安排、取消、请求身份，仍然各有自己的代码负责。

## 替身能让我看清什么

固定报告能帮助我调整列表和问题提示，却不能证明真实文件读取正确，也不能证明沙盒权限没问题。两份实现只是在输入输出的约定上接到同一个模型，不代表内部行为完全一样。

反过来，如果只想看结果页的间距，也没必要先完成真实扫描。之前这些事情都挤在同一个方法里，很容易让“调一下文字位置”变成“先找到一个合适的异常文件”。

接下来还剩一个容易踩的地方：这个 `SearchModel` 应该由谁创建、保存多久？闭包已经可以换了，但要是在 SwiftUI 的 `body` 里每次重新创建模型，刚才那份输入和任务又该归谁管。下一篇继续顺着它的生命周期往下看。
