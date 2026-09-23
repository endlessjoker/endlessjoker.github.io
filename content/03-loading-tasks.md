# 都写进 Task 了，怎么还会卡？

文件夹记住了，权限也拿到了，终于可以安心读正文了吧。

读取一个 Markdown 文件，核心甚至只有一行：

```swift
let text = try String(contentsOf: url, encoding: .utf8)
```

看着比前面保存书签省心多了。但这个调用要等文件读完、字符串解码完成才返回。文件小、磁盘快的时候，可能感觉不到等待；教材放在外接盘上，或者文件大一些，就不能继续靠“应该挺快的”过日子了。

阅读器还有侧栏、菜单、滚动条。我点开一篇文章，总不能让它们一起等着硬盘干活。

## 外面套个 Task，总可以了吧

假如把加载写成这样：

```swift
@MainActor
@Observable
final class ReaderModel {
    var markdown = ""
    var error: String?

    func select(_ url: URL) {
        Task {
            do {
                markdown = try String(contentsOf: url, encoding: .utf8)
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}
```

`select(_:)` 创建任务后就能返回，不用在这个函数里等正文读完。很容易因此觉得，耗时的事情已经交给后台了。

可这个 `Task` 写在 `@MainActor` 隔离的方法里，它的闭包会继承这里的隔离。等任务开始执行，读文件的同步调用仍然占着主线程。

只是把卡顿安排到了稍后，主线程并没有因此少干活。

第一篇里，`ReaderModel` 的 `@MainActor` 是为了管理界面状态。`markdown`、当前选中的文件、加载状态，都通过同一个隔离范围访问，避免不同并发任务随意同时修改它们。

这个安排管的是访问秩序，不会判断哪一行比较慢，然后热心地把它搬走。在这个 macOS 应用里，MainActor 上的工作由主线程执行。读文件占着它，界面也得等。

而 `Task` 表示一份异步工作，不是一条新线程。运行时负责安排任务在哪儿执行；创建了几个任务，并不意味着就开了几条专属线程。

这里要看的是 **Task 创建处的隔离上下文**。不能只凭“调用它的时候碰巧在主线程上”，就推断它一定继承 MainActor。

## 现在的阅读器把哪一段搬走了

项目里的 `select(_:)` 没有用上面那种写法。它先更新当前选择、恢复阅读位置，再启动加载。把历史记录等事情略去，读文件和回写正文的部分是这样：

```swift
readTask?.cancel()
selected = url
isLoading = true
isRendering = true

readTask = Task {
    let result = await Task.detached(priority: .userInitiated) {
        () -> Result<String, Error> in
        Result {
            let size = try url.resourceValues(forKeys: [.fileSizeKey])
                .fileSize ?? 0
            guard size <= 20_000_000 else {
                throw CocoaError(.fileReadTooLarge)
            }
            return try String(contentsOf: url, encoding: .utf8)
        }
    }.value

    guard !Task.isCancelled else { return }
    isLoading = false

    switch result {
    case .success(let text):
        markdown = text
        revision = UUID()
    case .failure(let issue):
        isRendering = false
        markdown = ""
        revision = UUID()
        error = "无法读取文件：\(issue.localizedDescription)"
    }
}
```

这里有两份任务。

外面的 `Task` 仍然继承 MainActor，负责协调加载、接收结果、修改界面状态。

里面的 `Task.detached` 没有继承这份 actor 隔离。查询文件大小、读取正文、解码字符串，都在里面完成，再把结果交出来。它用到的是这次选择时传入的 `url`，没有进去摸模型里随时可能变化的 `selected`。

`Result` 把成功读出的字符串或失败的错误装成一个值。所以外面等待 `.value` 时，拿到的是 `Result<String, Error>`，随后用 `switch` 分开处理。

读完以后也没有再写一遍 `DispatchQueue.main.async`。外层任务本来就在 MainActor 上，等待结束后，修改 `markdown` 的这段代码会继续在 MainActor 上执行。里面做过后台工作，不会把外层的隔离一起改掉。

顺便说，`.userInitiated` 是优先级，表示这件事来自当前操作。把它改成 `.background` 并不能替代 `detached`；给普通 `Task` 换个优先级，也不会让它摆脱原来的 actor 隔离。

## await 等着的时候，界面为什么还能动

这一行很容易读得心里打鼓：

```swift
let result = await worker.value
```

外面的任务不是还在 MainActor 上吗？现在又要等文件读完，岂不是绕了一圈，还是让主线程等？

区别在于怎么等。

如果结果还没出来，等待 `.value` 的任务可以挂起，把执行机会让出来。主线程不需要停在这里空等，其他界面工作可以继续执行。结果准备好以后，外层任务再恢复，接着往下走。

如果结果已经有了，也可能直接继续。`await` 标出的是一个可能挂起的位置，不是每碰到一次就必须停一次。

反过来，`String(contentsOf:)` 是同步调用。它读文件时并不会因为外面有一个 `Task`，就自动变成这种等待方式。把函数名改成 `loadAsync` 也没用，名字再努力，磁盘调用还是原来的磁盘调用。

现在这份实现把同步读取放进了 detached 任务，让它避开 MainActor。但它仍然会占用执行它的工作线程。以后如果同时扫很多本书、开很多次读取，就得考虑并发数量，以及是否需要专门承接阻塞文件操作的队列，不能靠不停创建 detached 任务解决。

20 MB 的文件大小检查能挡住一些过大的输入，也保证不了磁盘一定在多少毫秒内返回。

## 我都点第二篇了，第一篇就别抢着显示了

界面能在等待期间继续响应，新的事情也就能发生了。

比如先点 A，再点 B。假如 A 读得慢，完成顺序可能是这样：

```text
点 A → 开始读 A
点 B → 开始读 B
       B 先读完 → 显示 B
       A 后读完 → 又把正文换成 A
```

这时候侧栏选着 B，正文却是 A。每一步赋值都可以规规矩矩地在 MainActor 上执行，最后还是能显示错。

MainActor 不知道我已经不想看 A 了。它能约束访问，不能替应用决定哪个结果还算数。

项目在开始下一次读取前，先做了这件事：

```swift
readTask?.cancel()
```

旧任务拿到结果后，再检查自己是不是已经被取消：

```swift
guard !Task.isCancelled else { return }
```

于是 B 开始时取消 A；A 后来即使读完了，也不能继续把正文写回去。

这里检查的是 `Task.isCancelled`，也就是**正在执行这行代码的任务**。不能顺手改成检查模型属性里的 `readTask`：等 A 恢复时，那个属性已经可能指向 B 了。查着新任务的状态，决定旧结果能不能显示，越写越热闹。

检查的位置也有讲究。它在等待之后、修改 `isLoading` 和正文之前。如果只在开始时检查一次，后面那段等待期间发生的取消就漏掉了。

而且旧任务不该顺手把 `isLoading` 设成 `false`。B 可能还在加载，A 却回来宣布“都读完了”，连转圈都能一起搞错。

当前这段从取消检查到结果回写，中间没有新的 `await`。它作为一段 MainActor 上的同步代码执行，下一次点击不会插到这几行的正中间。以后往里面加异步调用，就得重新看看等待之后的状态还是否成立。

## cancel 了，文件真的不读了吗

正文不会被旧结果覆盖，听起来已经处理妥了。但第二篇末尾还留着一个问题：后台到底停没停？

看眼下的正文加载，`readTask` 保存的是外面那份任务。里面的 detached 任务虽然写在它的大括号里，却没有因此成为自动跟随它取消的结构化子任务。

所以调用 `readTask?.cancel()`，并不会自动取消里面的读取。外层仍然可能等到文件读完，再在 `guard` 那里丢弃结果。

这段代码做到了“旧结果别再显示”，还不能据此说“旧的磁盘读取已经停止”。

同一个项目的目录扫描，取消就多接了一步。它把里面的任务保存成 `worker`，等待时加上取消处理：

```swift
let worker = Task.detached(priority: .userInitiated) {
    Self.scan(url, directory: library.isFolder)
}

let result = await withTaskCancellationHandler {
    await worker.value
} onCancel: {
    worker.cancel()
}

guard !Task.isCancelled else { return }
```

外层被取消时，`onCancel` 把取消信号转给 `worker`。扫描目录的循环里也确实检查了 `Task.isCancelled`，看到取消就退出循环。

两边得接得上。只调用 `worker.cancel()`，里面完全不理会，也不会凭空停下来。

取消是一个需要代码配合处理的信号，不是强行掐断线程。即使补上转发，一次已经开始的同步文件读取，也不会自动在读到一半时被中断。目录扫描是在文件之间检查，遇上一次正在进行的磁盘调用，仍然可能要先等它返回。

这也解释了为什么上一章不能把“我调用了 cancel”当成“所有读取都已结束”，然后立刻认定访问授权已经没人用了。

现在能看清接下来该改哪里了：正文加载可以补上取消转发和开始读取前的检查，减少已经作废却还没开始的工作；如果要在移除目录时严谨地结束访问，就还得协调正在使用它的任务何时真正完成。一个 `cancel()` 没法把这些事全包办了。

读文件这条线到这里，终于从“包个 Task 应该就行”，变成了能逐行看出事情发生在哪里。

不过正文变成字符串，离屏幕上出现排好版的文章还差一段。`markdown` 已经赋值了，`WKWebView` 怎么知道要重画？总不能每改一下阅读进度，就顺手把整篇文章重新加载一遍。下一个要看的，就是 SwiftUI 和这块网页之间怎么传话。
