# weak 写上了，谁还在留着对象？

上一章把 SwiftUI 和网页接起来以后，Coordinator 里还剩一句值得单独翻出来的代码：

```swift
weak var web: WKWebView?
```

它得调用网页里的 JavaScript，所以需要找到 `WKWebView`。但只是为了调用几个方法，为什么又要加一个 `weak`？

顺着初始化往下看，这个网页也在留着 Coordinator。两边都说“我还要用你”，事情就有意思了。

## 问号不负责放手

假如这里写的是普通的可选属性：

```swift
var web: WKWebView?
```

它仍然会强引用赋进去的网页对象。`?` 只说明这个位置可以没有值，不表示“有值时就轻轻拿着”。

给这个属性赋 `nil`，也只是断开它自己的引用。其他地方还在持有，网页就继续活着。ARC 不会因为其中一个变量说用完了，就把其他人手里的对象一起收走。

项目注册回调时，有这两句：

```swift
config.userContentController.add(context.coordinator, name: "reader")
config.setURLSchemeHandler(context.coordinator, forURLScheme: "reader-local")
```

第一句让 Coordinator 接收网页消息；第二句让它处理正文里的本地资源请求，比如图片。

注册也会建立持有关系。省掉 WebKit 内部的配置对象，只画这两条路，大致是这样：

```text
WKWebView
  ├─ 消息处理器注册 ───→ Coordinator
  └─ 自定义资源处理器 ─→ Coordinator

Coordinator ── weak ──→ WKWebView
```

上面两条路径会留住 Coordinator，下面这条弱引用不会留住网页。

如果把下面也改成强引用，就绕回来了：网页通过注册的处理器留着 Coordinator，Coordinator 又留着网页。外面的视图即使不再需要它们，这一圈也有可能让它们继续互相持有。

ARC 会数强引用，但不会顺手判断“这几个对象已经没人要了，我帮它们拆个环”。“自动”两个字，业务还没包那么广。

`weak` 改的就是回去的这条线。等网页的其他强引用都结束，Coordinator 不会凭这个属性阻止它释放；网页释放后，弱引用会变成 `nil`。

这也解释了调用时的 `web?.evaluateJavaScript(...)`：网页可能已经不在了，那就不用再发命令。

换成 `unowned` 也不负责留住对象，但它要求使用时对象仍然有效。这里本来就允许网页先结束，为了少处理一个可选值，给自己加这个保证，没必要。

## 名字都像回调，持有方式却不一样

初始化里还有一句：

```swift
web.navigationDelegate = context.coordinator
```

`navigationDelegate` 本身是弱引用。不能看见它也叫 delegate，就把它和消息处理器画成同一种线。

同样，Coordinator 的 `let model: ReaderModel` 是正常的强引用。它处理进度和排版结果时需要访问模型；当前模型没有再持有这个 Coordinator，并不能仅凭这一句就认定有环。

所以我得沿着属性和注册关系往回找，看看能不能真的绕回来。把类里的引用统统改成 `weak`，很可能只是把“对象不走”换成“对象提前没了”。功能照样坏，方向还挺灵活。

## 页面不用了，就在这里收尾

第四篇出现过这段拆除代码：

```swift
static func dismantleNSView(_ web: WKWebView, coordinator: Coordinator) {
    web.configuration.userContentController
        .removeScriptMessageHandler(forName: "reader")
    web.navigationDelegate = nil
    coordinator.loadingTask?.cancel()
}
```

现在再看，它没有在执行一个统一的“释放全部”按钮，而是在结束几件具体的事。

移除消息处理器，结束的是名为 `reader` 的注册。把 `navigationDelegate` 置空，是解除导航回调关系；这个属性原本就是弱引用，不用把这一行说成拆掉了某个强引用环。

而移除消息处理器，也不能直接推出 Coordinator 当场释放。自定义资源处理器的注册、SwiftUI 对协调对象的管理，都还可能持有它。当前网页和相关配置结束后，这些持有关系才会陆续结束。

这里主动收尾，是因为底下的视图已经要拆除了。没必要等对象彻底没人引用，才想起来取消消息和任务。

要是本来就有一个强引用环，又打算在 `deinit` 里拆掉它，那就尴尬了：恰恰是这个环，让 `deinit` 没机会执行。

而且“当前文章不显示了”和“这份网页被拆除了”也不是同一回事。上一章已经让网页复用了，普通的章节切换不会每次都走 `dismantleNSView`。每轮排版的收尾，还得由每轮排版自己管。

## weak 也不会替任务点取消

Coordinator 会给排版安排一个超时任务。当前的 `watchLoading()` 是这样：

```swift
func watchLoading() {
    loadingTask?.cancel()
    loadingTask = Task { [weak self] in
        try? await Task.sleep(for: .seconds(12))
        guard !Task.isCancelled, let self else { return }
        self.fail("阅读引擎未及时响应，请重新加载。")
    }
}
```

创建新任务之前，先取消上一轮。新任务等待 12 秒，醒来后如果还有效，就报告超时。

这里的 `[weak self]` 让任务在等待期间不必强留着 Coordinator。如果协调对象已经结束，醒来时拿不到它，直接返回。

但排版成功时，Coordinator 通常还活着，它后面还要处理滚动和下一篇文章。弱引用自然也还有效。

这时如果只靠 `weak`，任务仍然可以在 12 秒后醒来，给早就显示出来的文章报一个超时。文章读得好好的，阅读器突然说自己没响应，这就很有节目效果。

所以收到 `rendered` 消息时，项目会取消这份任务；视图拆除时，也会取消。`weak` 处理对象是否被留住，取消处理这轮工作是否还需要继续。

后面的 `Task.isCancelled` 同样不能省。`Task.sleep` 遇到取消会抛出错误，但前面的 `try?` 把错误吞掉了，控制流还会往下走。少了这道检查，“排版成功，取消计时”反而可能立即执行超时处理。

另外，把 `loadingTask` 直接设成 `nil` 也不等于取消任务。那只是丢掉了用于管理它的句柄，运行中的任务不会因此自动停止。

## guard let self 放在哪儿，差别还真不小

上面的代码先等待，再解包 `self`。

如果只是为了后面少写几个问号，把解包挪到最前面：

```swift
loadingTask = Task { [weak self] in
    guard let self else { return }
    try? await Task.sleep(for: .seconds(12))
    guard !Task.isCancelled else { return }
    self.fail("阅读引擎未及时响应，请重新加载。")
}
```

一旦成功解包，局部的 `self` 就是强引用。后面还要调用它的 `fail`，因此这份引用会跨过中间的等待，把对象留住。

闭包入口的弱捕获还在，后面却又取得了一份强引用。只搜代码里有没有 `[weak self]`，看不出这层区别。

不过这也不能直接叫“永久泄漏”。这份超时任务正常会结束，任务执行完会释放执行闭包所捕获的引用。就算属性里还存着已完成的任务句柄，也不意味着它永远保留执行时的整个闭包。

一次确实需要用到对象的有限工作，让对象活到工作结束，本来就可能是合理的。这里选择晚一点解包，是因为旧网页的超时报告没有必要反过来延长协调对象的生命。

## 换成一直运行的循环，就不能这么随意了

阅读器还有一个监看文件变化的 `watcher`，每隔两秒检查当前文档，隔一段时间再扫描目录。它和前面的超时任务不同，并没有“等十二秒就自然结束”这一说。

把监看简化为每轮调用一次 `reload()`，再假设把强引用拿在循环外面，代码会是这样：

```swift
watcher = Task { [weak self] in
    guard let self else { return }
    while !Task.isCancelled {
        do {
            try await Task.sleep(for: .seconds(2))
        } catch {
            return
        }
        self.reload()
    }
}
```

一旦进入循环，任务就一直需要这份 `self`。如果又准备等模型的 `deinit` 来取消任务，就可能互相等：模型等任务放手，任务等模型释放时喊停。开头那个 `weak` 还在，看着甚至挺让人放心。

当前实现把 `guard let self` 放在循环里面、每次睡眠之后。它需要模型时才取得强引用，处理完这一轮，不把同一份强引用一直带着进入下一轮睡眠。

但“一轮”里面也可能再次 `await`。比如扫描目录时，后面还要把结果写回模型，这一轮取得的强引用就会跨过扫描等待。不能因为 guard 在循环里面，就断言模型在任何等待期间都完全没人留着。

顺着这里看，也能发现当前代码还可以补的一处：监看任务的睡眠用了 `try?`，醒来后只检查 `self`，没有像排版超时那样再检查取消。如果睡眠中被取消，模型又仍然活着，就可能多做这一轮工作。

如果继续改这段，睡眠后应先挡住取消：

```swift
guard !Task.isCancelled, let self else { return }
```

这属于对现有代码的改进点，还不能写成已经处理好了。至于目录扫描启动后怎么停、什么时候才算完全停下来，还要接上第三篇的任务取消关系。

现在再看这些 `weak`，就得把后面的代码也一起看完。到底有没有重新取得强引用，跨过了哪段等待，工作靠什么结束，比闭包开头那几个字更能说明问题。

下一处想顺着看的是全文搜索。每输入一个字都从头扫教材，机器愿意干，我还嫌它太勤快。项目里那 300 毫秒的等待，正好能把前面几篇的任务、取消和旧结果串起来。
