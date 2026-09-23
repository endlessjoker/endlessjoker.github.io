# SwiftUI 一更新，网页就得重载吗？

正文读出来了，已经放进 `ReaderModel.markdown`。如果只是显示一段文字，后面接个 `Text` 就能收工。

但教材还有代码高亮、公式、图片和段落定位。`epubBooks` 把这些排版交给了 WebKit，SwiftUI 负责外面的窗口和工具栏。

于是问题变成了：我改的是 Swift 里的字符串，网页里的正文怎么跟着变？反过来，网页滚到一半，窗口底下那个阅读百分比又怎么知道？

两边都挺忙，只是还没互相认识。

## 这个 struct 和网页不是同一个东西

项目用 `ReaderWebView` 把 `WKWebView` 放进 SwiftUI。它遵循 `NSViewRepresentable`，也就是把一个 AppKit 视图接进 SwiftUI 的接口。

`ReaderWebView` 是 struct，描述现在需要怎样的阅读界面。`WKWebView` 则是有自己状态的对象：页面加载到哪了、滚动到哪了、选中了哪段文字，都在它那边。

SwiftUI 重新计算 `body`、构造新的 `ReaderWebView` 值，不等于每次都新建一个 `WKWebView`。视图身份保持时，底下的对象可以继续复用。

创建这份对象的地方是 `makeNSView`。项目在这里配置消息处理器、创建 `WKWebView`，然后加载应用里打包好的 `Reader.html`。后续更新走 `updateNSView`：

```swift
func updateNSView(_ web: WKWebView, context: Context) {
    context.coordinator.update()
}
```

假如图省事，在这个方法里每次都调用 `loadHTMLString`，SwiftUI 只要安排一次更新，网页就得重新加载一遍。

我可能只是改了一下字号，它却开始重新建页面、跑脚本、恢复滚动位置。功能最后也许能对上，但干的活明显太多了。

现在的 `Reader.html` 更像一份常驻的排版程序。它带着样式和 JavaScript，切换文章时，把新正文交给里面的 `renderDocument`；调整字号时，调用 `setFontSize`。不必为了换一篇文章，把这份程序也重新加载一次。

## Coordinator 记着上次交代了什么

`updateNSView` 里面又多绕了一层 `Coordinator`。起初看这个名字，很容易以为是 SwiftUI 规定必须加的一层架构。

其实普通的单向展示不一定需要它。这里用得上，是因为网页既要接命令，又要往回报告事情，还得记住上一次已经处理了哪些变化。

项目创建它的代码很短：

```swift
func makeCoordinator() -> Coordinator {
    Coordinator(model: model)
}
```

SwiftUI 会在创建底下的 AppKit 视图前，先创建这份协调对象，之后通过 `context.coordinator` 交回来。它的生命周期跟着这份被管理的视图，不需要随着每个新的 `ReaderWebView` 值重新生成。

Coordinator 里有这些属性：

```swift
let model: ReaderModel
weak var web: WKWebView?
var ready = false
var revision: UUID?
var font: Double = 0
```

`model` 是当前阅读模型，`web` 指向它要操作的网页。`revision` 和 `font` 留着上一次发送过的正文标记和字号。

模型里的字号表示“现在想用多大”，Coordinator 里的字号表示“上次已经告诉网页用多大”。两个值看着重复，问的却是两件事。

这份 Coordinator 持有的是创建时传进来的模型。当前阅读窗格一直使用同一份模型，所以成立；以后如果在相同视图身份下直接换一个 `ReaderModel` 对象，就得同步更新 Coordinator 的引用，不能指望它自动换人。

## SwiftUI 怎么知道这里有变化

外面的 `ReaderPane` 把需要的值传了进来：

```swift
ReaderWebView(
    model: model,
    revision: model.revision,
    fontSize: model.fontSize,
    searchRequest: model.searchRequest,
    jumpRequest: model.jumpRequest,
    isLoading: model.isLoading,
    notesRevision: model.notesRevision,
    captureRequest: model.captureRequest
)
.id(model.webSession)
```

这里在 `body` 中读取了可观察模型的属性，SwiftUI 就能跟踪相应依赖，把变化带到视图更新里。

所以虽然 `updateNSView` 最后只调用 `coordinator.update()`，前面这串输入也有作用：需要跟随哪些状态变化，在调用处就能看见。

但这不是给 `updateNSView` 规定调用次数。不能认定它“每换一篇正文才来一次”，然后放心地把重活都塞进去。它来了以后，还得自己判断有什么值得做。

## 正文和字号各管各的

把标注、搜索和跳转先略去，Coordinator 的更新逻辑是这样：

```swift
func update() {
    guard ready, !model.isLoading else { return }

    if revision != model.revision {
        revision = model.revision
        watchLoading()
        call("renderDocument", [
            "markdown": model.markdown,
            "path": model.selected?.path ?? "",
            "fontSize": model.fontSize,
            "progress": model.progress
        ])
        font = model.fontSize
    }

    if font != model.fontSize {
        font = model.fontSize
        call("setFontSize", font)
    }
}
```

完整代码还把段落位置、锚点等数据传给网页。这里只保留正文、路径、字号和滚动比例，方便看清什么时候发哪条命令。

第三篇读完文件后，有一句 `revision = UUID()`。它表示正文有一轮新内容要交给网页，Coordinator 看见标记变了，才调用 `renderDocument`。

这个 UUID 不负责比较 Markdown，也没有“越大越新”的顺序。它只是一次明确的通知：这批正文需要处理。

字号变了，正文标记没变，就只调用 `setFontSize`。网页里对应的函数甚至只有一行实际操作：

```javascript
function setFontSize(size) {
    document.documentElement.style.setProperty('--font-size', size + 'px');
}
```

改 CSS 后，浏览器仍然可能重新排版。不过 Markdown 不用重新解析，整份页面也不用重新加载，已经省掉了一大圈。

发送新正文时，参数里已经带上了字号，所以代码紧接着把 `font` 也记下来。否则下面又检查到字号不一致，还得多发一遍同样的设置。

`call` 是项目自己写的辅助方法，最后通过 `evaluateJavaScript` 调用网页函数。参数会先经过 JSON 序列化，正文里的引号、换行不用自己拼成 JavaScript 字符串。读的是程序教材，里面到处都是引号，手工拼起来简直是在主动给自己找事。

这里记录 `revision` 的时机，是准备发送命令的时候。它表示已经派发过这轮正文，不代表网页已经成功排好了。成功与否，还得等另一边回话。

## 网页加载好了，正文不一定排好了

更新开头有两个条件：

```swift
guard ready, !model.isLoading else { return }
```

`ready` 管排版程序有没有准备好，`isLoading` 管 Swift 这边是不是还在读文件。

刚创建 `WKWebView` 时，调用 `loadHTMLString` 并不会让所有页面脚本立即准备就绪。项目等到导航完成回调，再打开这道门：

```swift
func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    ready = true
    update()
}
```

正文先读完，就等网页；网页先准备好，就等文件。两个条件都满足后，再把当前正文发过去。

但 `didFinish` 对应的是这份页面的导航完成。后面调用 `renderDocument`，解析 Markdown、处理公式、等字体和图片、恢复滚动位置，是另外一段工作。

所以模型里保留了两个状态。`isLoading` 在文件读取完成后结束；`isRendering` 还要等网页报告这一轮排版流程结束，才会关掉加载提示。

网页会发回这样一条消息：

```javascript
send({ kind: 'rendered', path: currentPath });
```

这里的“完成”是阅读器自己规定的阶段，也不是保证所有图片都加载成功。当前实现会等待图片加载、失败或者超时，然后继续恢复位置，免得一张坏图让整篇文章永远转圈。

Swift 收到 `rendered` 后，会核对文档路径、取消超时任务，再把 `isRendering` 设成 `false`。

这里还接着第三篇的旧结果问题：如果已经切到 B，A 发回来的完成消息不能替 B 宣布结束。路径检查能区分不同文档；网页端另外用 `generation` 标记排版轮次，异步等待后发现自己过期，就放弃后面的定位和完成通知。

## 滚动条动了，消息再传回来

网页往 Swift 发消息，走的是创建视图时注册的这个入口：

```swift
config.userContentController.add(context.coordinator, name: "reader")
```

JavaScript 里的 `send` 对应它：

```javascript
const send = body => window.webkit?.messageHandlers.reader.postMessage(body);
```

名字都是 `reader`，两边才接得上。`Coordinator` 遵循 `WKScriptMessageHandler`，在 `userContentController(_:didReceive:)` 里接收这些消息。

滚动时，网页发送 `progress`、文档路径、滚动比例和段落位置。Swift 根据消息的 `kind` 选择处理分支，再把数据交给模型。

`saveProgress` 里最关键的是这两行：

```swift
guard document == selected?.path else { return }
progress = min(1, max(0, value))
```

先确认是当前文档，再把比例限制在 0 到 1。后面还会保存进度，供下次打开时恢复。

于是这条路终于接通了：

```text
网页滚动
  → JavaScript 发 progress 消息
  → Coordinator 收到消息
  → ReaderModel.progress 改变
  → SwiftUI 更新阅读百分比
```

关键是，保存进度没有顺手修改正文的 `revision`。即使这次状态变化又引起一轮 SwiftUI 更新，也不会仅仅因为进度变了，就再发一次 `renderDocument`。

否则就可能变成：滚一下、报告进度、重排正文、恢复位置、再报告进度。只是想往下翻两行，双方却热情地互相通知个没完。

## 真需要重建时，再换身份

前面的调用里还有一句 `.id(model.webSession)`。

它没有绑正文的 `revision`，更没有现场写一个 `.id(UUID())`。如果每次计算界面都换身份，就没法指望底下那份网页安稳复用了。

`webSession` 平时保持不变。阅读引擎出错，点击“重新加载”时，模型才执行：

```swift
func retryRenderer() {
    renderFailure = nil
    isRendering = true
    webSession = UUID()
}
```

身份换掉，SwiftUI 会结束原来的那份视图，创建新的网页和 Coordinator，再走初始化与加载过程。

`revision` 用来通知当前网页换正文，`webSession` 用来换掉网页本身。现在再看到两个 UUID，就不觉得只是多写了一份随机数了。

旧视图离开时，项目也留了收尾：

```swift
static func dismantleNSView(_ web: WKWebView, coordinator: Coordinator) {
    web.configuration.userContentController
        .removeScriptMessageHandler(forName: "reader")
    web.navigationDelegate = nil
    coordinator.loadingTask?.cancel()
}
```

消息处理器撤掉，导航回调解绑，等候排版的超时任务也取消。

这几行倒是又引出了一个问题：前面那个 `web` 属性为什么是 `weak`？消息处理器注册进去以后，谁在留着 Coordinator？要是只等 `deinit` 再收尾，会不会压根等不到？

本来只是塞了个网页，结果连谁负责留住谁，也得算清楚。下一篇得把这几根线画出来。
