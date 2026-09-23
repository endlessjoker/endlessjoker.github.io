# 搜索框不大，状态还挺多

上一篇看完搜索的防抖，事情还没结束。

列表里留着 `actor` 的结果，输入框已经改成了 `Task`。这时候点一下旧结果，打开的文件和行号属于 `actor`，准备高亮的词却从输入框里拿了 `Task`。

顺着代码能走到这个组合。搜索任务没有写错结果，点击按钮时才把两轮信息拼到了一起。

现在模型用三个属性记录全文搜索：

```swift
var fullQuery = ""
var fullResults: [BookSearchResult] = []
var searching = false
```

每个属性都挺合理。词在这里，结果在这里，忙不忙也在这里。但 `fullResults` 是用哪个词搜出来的，没地方记。

当然可以再加一个 `resultsQuery`。只是每次替换结果时，得记得同时替换它；清空时，又得一起清空。后面再加等待状态、错误提示，搜索框没长大，需要彼此照应的变量倒是越来越多。

我想保留搜索期间的旧列表，点开时仍然使用旧列表自己的查询词。下面先把这个改法写出来，阅读器目前还用着原来的三个属性。

## 一批结果，连同来历一起存

先给“一次搜索”和“这次得到的一批结果”各留一个类型：

```swift
struct SearchRequest: Sendable {
    let id = UUID()
    let query: String
}

struct SearchBatch: Sendable {
    let request: SearchRequest
    let results: [BookSearchResult]
}
```

`SearchRequest` 记下这次真正拿去扫描的词，也就是去掉首尾空白之后的那个值。`id` 区分每次发起的搜索，后面再用。

`SearchBatch` 把请求和结果放在一起。拿到这批结果，就能知道它对应哪个词，不用回头打听输入框现在写了什么。

两个 struct 的字段都是 `let`。组装好以后，不能单独把这批结果的词换掉。新搜索完成，就创建另一批。

它们仍然可以装错数据。把搜 `actor` 的数组和搜 `Task` 的请求传进同一个初始化方法，编译器不会读懂摘要然后报警。这个类型提供了放在一起的位置，正确配对还得在结果返回时做。

## 忙不忙，不能把所有事情都说清楚

`searching` 为 `true` 的时候，可能正在等那 300 毫秒，也可能已经开始读文件。

它为 `false` 的时候，可能还没搜过，也可能搜完了一条都没找到。

如果这些区别要显示在界面上，光改这个 Bool 已经不够用了。我准备让搜索状态直接带上它需要的数据：

```swift
enum SearchState {
    case idle
    case waiting(SearchRequest, previous: SearchBatch?)
    case searching(SearchRequest, previous: SearchBatch?)
    case finished(SearchBatch)
}
```

`waiting` 带着这次准备搜索的请求，以及可以继续显示的旧结果。第一次搜索没有旧结果，所以 `previous` 可以是 `nil`。`searching` 也是这两份东西，只是工作已经开始。

`finished` 必须带回一批结果。哪怕里面的数组是空的，也留下了“搜过什么”的记录。它和 `idle` 的区别终于不用靠猜了。

这些括号里的数据就是枚举的关联值。它们随着某一次具体的状态保存下来：第一次 `.waiting` 可以等 `actor`，下一次可以等 `Task`。这和给一个枚举项固定写上字符串原始值是两回事。

一个 `SearchState` 值在某个时刻只会是其中一种情况。现在没法同时把“正在等”和“已经完成”都设成 `true`，也没法只写一句 `.finished` 却不交代结果。

不过“正在搜新词，同时展示旧结果”是我想要的情况，所以明确放进了 `previous`。把它误判成非法状态，再用枚举消灭掉，界面就只能每次输入都清空了。类型怎么写，还是得先想好这个搜索框要怎么用。

## 列表应该显示哪一批

旧结果可能挂在等待或搜索状态里，新结果则在完成状态里。界面没必要每次都把这段选择再写一遍，可以让枚举自己提供：

```swift
extension SearchState {
    var displayedBatch: SearchBatch? {
        switch self {
        case .idle:
            return nil
        case .waiting(_, let previous), .searching(_, let previous):
            return previous
        case .finished(let batch):
            return batch
        }
    }

    var pendingRequestID: UUID? {
        switch self {
        case .waiting(let request, _), .searching(let request, _):
            return request.id
        case .idle, .finished:
            return nil
        }
    }
}
```

`let previous` 和 `let batch` 把对应情况里保存的数据取出来，`_` 表示这一处用不上那份数据。

这两个是计算属性，没有另存一个结果数组或请求编号。状态变了，下次读取就按新状态计算，因此也不用在某个分支里记得“顺便更新 displayedBatch”。

这里把四种情况都写出来了，没有用 `default` 兜底。将来真加了失败状态，编译器会指出这些 switch 还没处理它。这样至少不会因为一个宽松的默认分支，让新状态悄悄显示成“还没开始”。

模型中保留输入框的值，把另外两个属性换成状态：

```swift
var fullQuery = ""
private(set) var searchState: SearchState = .idle
```

`private(set)` 让界面能读状态，但不能随手把它改成完成。输入框照常绑定 `fullQuery`，搜索方法负责推进 `searchState`。

`ReaderModel` 本来就用了 `@Observable`。界面求值时读取 `model.searchState.displayedBatch`，也会读到模型的 `searchState`，因此状态更新可以触发依赖它的界面更新。这个 enum 不需要再套一层观察宏。

## 状态不会自己往下走

枚举只规定有哪些值，没有规定它们按什么顺序出现。等待、开始、完成，还得由 `searchBooks()` 来安排。

我打算把上一版的 detached 扫描和取消转发收进 `scanBooks(sources:query:)`。这个辅助方法返回结果数组，内部仍然由 worker 读文件；函数签名里有 `async`，本身可不会把同步读取搬离 MainActor。

这样搜索方法就可以专门看清这些状态是在什么时候改的：

```swift
func searchBooks() {
    fullSearchTask?.cancel()

    let query = fullQuery.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !query.isEmpty else {
        searchState = .idle
        return
    }

    let request = SearchRequest(query: query)
    let previous = searchState.displayedBatch
    let sources = libraryChapters.flatMap { id, chapters in
        chapters.map { (id, $0.url) }
    }
    searchState = .waiting(request, previous: previous)

    fullSearchTask = Task {
        try? await Task.sleep(for: .milliseconds(300))
        guard !Task.isCancelled,
              searchState.pendingRequestID == request.id else { return }

        searchState = .searching(request, previous: previous)
        let results = await scanBooks(sources: sources, query: request.query)

        guard !Task.isCancelled,
              searchState.pendingRequestID == request.id else { return }

        searchState = .finished(SearchBatch(request: request, results: results))
    }
}
```

`previous` 在替换状态前取得。假设 `actor` 已经搜完，接着输入 `Task`，等待期间又改成 `TaskGroup`，新请求会继续带着原来那批 `actor` 结果，不会把还没完成的 `Task` 当成一批结果。

最后的赋值把这轮任务捕获的 `request` 和返回的 `results` 放进同一个 batch。这就是前面提到的配对位置。输入框后来怎么变，都不用再去读它。

完成之后也不用补一句 `searching = false`。界面已经能从 `.finished` 知道这次工作结束了。少一个开关，就少一次“结果都出来了，怎么还在转”的机会。

清空输入时，取消旧任务并换成 `.idle`，显示中的结果也随之消失。这里的空闲说的是当前界面没有待完成的请求；已经取消的后台读取仍然可能在收尾。

## 同一个词，也可能搜了两次

前面那个 UUID 终于有事情做了。

如果只靠查询词判断旧结果能不能回来，会遇到这种顺序：

```text
请求 A：actor
请求 B：Task
请求 C：actor
```

等 A 回来时，当前又在搜 `actor`。只比较字符串，A 和 C 看着完全一样。但两次发起之间，文件可能改过，待扫描的目录也可能变过，A 不能冒充 C。

请求各有编号，回来时就比较自己是不是模型当前等待的那一份。清空输入以后，`pendingRequestID` 是 `nil`，旧请求自然也对不上。

上一版在这个入口里正确取消旧任务、检查取消，已经能挡住旧任务回写。这次保留取消检查，又把请求身份直接记进状态，是为了把“现在到底在等谁”也表达出来。

编号检查不会让旧扫描停止。取消和扫描中的检查仍然要留着，否则旧任务只是忙完了不能显示，文件还是白读了一遍。

这些状态赋值仍由 MainActor 管理。每次等待之后都重新核对身份，核对与紧接着的赋值之间没有新的 `await`。MainActor 负责让这些同步操作不被同一 actor 上的其他任务插进来，请求编号负责判断回来的是否还是当前这一轮。

一个管执行时机，一个管事情有没有过期。光写上 MainActor，旧请求也不会自动变懂事。

## 点开时，跟着这一批走

界面可以直接按状态显示提示，再取出当前可展示的 batch：

```swift
VStack {
    switch model.searchState {
    case .idle:
        Text("还没有开始搜索")
    case .waiting(let request, _):
        Text("等待搜索“\(request.query)”")
    case .searching(let request, _):
        ProgressView("正在搜索“\(request.query)”")
    case .finished(let batch):
        Text("“\(batch.request.query)”：\(batch.results.count) 条结果")
    }

    if let batch = model.searchState.displayedBatch {
        Text("下方结果来自“\(batch.request.query)”")
        List(batch.results) { result in
            Button {
                model.openResult(result, from: batch)
            } label: {
                Text(result.snippet)
            }
        }
    }
}
```

这里保留了两个词各自出现的位置。上面可以写“正在搜索 Task”，下面仍然标着“结果来自 actor”。旧列表继续留着，但不再装作自己已经更新过。

按钮闭包用的 `result` 来自这一批的数组，同时把这一批也传进去：

```swift
func openResult(_ result: BookSearchResult, from batch: SearchBatch) {
    search = batch.request.query
    activate(result.library, document: result.url)
    pendingFind = batch.request.query
    searchResultLine = result.line
}
```

这次不碰 `fullQuery`。点击 `actor` 的旧条目，就按 `actor` 打开和查找。

空结果也没有被当成“从未搜索”。`.finished` 带着零条结果，仍然能显示这轮搜了什么。不过文件过大、读取失败会跳过，最多收集 300 条，这些扫描限制还在；换个状态类型并不能让“零条”变成“所有文件都查过了”。

## 先别给它起一个很大的名字

写到这里，只多了几个具体类型，搜索方法却还是那个搜索方法。暂时不需要为了它搭一个通用状态机框架。

这个 enum 也没有替我规定所有合法的转换。模型内部仍然能从 `.idle` 直接赋成 `.finished`，也能故意把不配套的请求和结果塞在一起。它省掉的是一部分容易组合错的零散属性，剩下的关系在几处明确的赋值里核对。

失败状态先不硬凑。当前扫描遇到某个文件读不了，会跳过它并继续搜其他文件。直接加一个 `.failed`，还得回答：一本书读不了，其他九本的结果要不要留？搜到上限提前返回，算完成还是失败？

这些问题得等扫描方法把具体情况带回来，界面才有东西可说。下一篇就接着查这些被 `try?` 忽略掉的错误。搜索框看起来挺安静，也可能只是模型没把发生的事告诉我。
