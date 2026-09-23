# 都放进 actor 了，怎么还会重复加载？

阅读器打开一个教材目录，要扫一遍 Markdown 文件；过一阵子还会再扫，看看文件列表有没有变化。这两处扫描都已经在后台任务里做，主界面不用陪着遍历目录。

当前代码有两条入口。`scanLibrary(_:)` 给某个阅读位置启动扫描，先取消它自己保存的上一次任务；`startWatching()` 每隔一段时间也会启动一次扫描。后者没有复用前者的 `scans` 字典。如果时间正好撞上，同一个目录可能被扫两遍。我没测到它实际撞上的次数；只是从两条调用路径能看出这种可能性。

第一反应很容易是：做一个扫描 actor，不就能把它们排队了吗？这句话只对了一半。actor 能保护那份“正在扫描”的记录，但光有 actor，还不能让两个请求自动共用同一次工作。

## 一把锁，锁不住一整段 await

假设我把扫描结果放进一个 actor：

```swift
actor NaiveChapterIndex {
    private var cache: [ScanRequest: [Chapter]] = [:]

    func chapters(for request: ScanRequest) async -> [Chapter] {
        if let saved = cache[request] { return saved }

        let result = await Task.detached {
            ReaderModel.scan(request.url, directory: request.directory)
        }.value

        cache[request] = result
        return result
    }
}
```

看着很稳：先查缓存，没有就读盘，读完存起来。`cache` 也只有 actor 自己能碰。

问题出在 `await`。请求 A 查到缓存为空，启动扫描，然后等待。它等的时候，actor 可以去处理请求 B。B 查缓存，还是空的，于是也启动一份扫描。两份扫描都能完成，再分别回来写缓存。actor 没有让字典发生数据竞争，但它也没许诺“从查空到写入，中间不许别人进来”。

这就是重入。actor 一次只执行一段隔离代码；走到可能挂起的位置，另一份工作可以进来。`await` 前成立的判断，`await` 后未必还成立。把数据库连接、网络请求、文件扫描放进去，道理都一样。actor 不是给异步函数从头到尾包了一个巨大的互斥锁。

这里的坏处不只是多干一遍活。如果 A 和 B 读到的目录状态不同，晚完成的那份未必更新，却可能最后写进缓存。只有“不发生数据竞争”这个保证，还不足以决定哪个结果有效。

这段是我假设要新增缓存时容易写出的版本。个人阅读器当前没有 `NaiveChapterIndex`，也没有这个缓存。

## 等待之前，先把“有人在做”记下来

这次不急着缓存结果。先只解决**同时要同一份扫描**的问题：

```swift
nonisolated struct ScanRequest: Hashable, Sendable {
    let url: URL
    let directory: Bool
    let generation: UInt64
}

actor ChapterScans {
    private struct Flight {
        let token: UUID
        let task: Task<[Chapter], Never>
    }

    private var inFlight: [ScanRequest: Flight] = [:]

    func chapters(for request: ScanRequest) async -> [Chapter] {
        let flight: Flight

        if let running = inFlight[request] {
            flight = running
        } else {
            let worker = Task.detached(priority: .utility) {
                ReaderModel.scan(request.url, directory: request.directory)
            }
            flight = Flight(token: UUID(), task: worker)
            inFlight[request] = flight
        }

        let result = await flight.task.value
        if inFlight[request]?.token == flight.token {
            inFlight.removeValue(forKey: request)
        }
        return result
    }
}
```

关键就是 `inFlight[request] = flight` 位于第一个 `await` **之前**。请求 A 把任务句柄放进去，然后才去等。A 挂起后 B 进来，会拿到同一个句柄，等同一份结果。`Task` 在这里不是缓存的文件列表，而是一张“这份工作已经有人接了”的票。

两个等待者都返回时会各自尝试清理字典。第一个删掉，第二个可能在稍后才恢复；这期间新请求可能已经放入下一张票。所以清理前还要对一下 `token`，只能删自己等的那一轮。没有这个检查，旧等待者回来得晚，反而会把新扫描从字典里擦掉。这个小 actor 只合并**重叠**请求，不保存长期缓存。

`generation` 是“这一轮想看哪版目录”的编号。同一路径也可能需要重新扫描：定时检查到了下一轮，或者用户明确要求刷新，不能因为 URL 一样就永远拿旧活。想共用工作，就传同一个编号；想要新一轮，就换编号。实际接线时，编号得由发起扫描的那一层统一管理，随手在每个调用处各造一个随机数，合并就又成了摆设。`ScanRequest` 标成 `nonisolated`，是因为这个工程的默认隔离设为 MainActor，而这份键值要在独立 actor 里比较和哈希。

这里还要留意，`ReaderModel.scan` 本身是同步遍历文件。我把它放进 detached task，等待它的 `.value` 才会挂起；把同步扫描直接写在 actor 方法里，actor 在遍历期间仍要执行这段代码，`async` 函数名不会把磁盘读取变成可挂起的异步 I/O。

## 共用任务以后，取消不再是私事

现有 `scanLibrary(_:)` 的规则是：同一个位置有新扫描，就取消自己保存的旧任务。它专门管理自己的那条扫描链。

如果改成上面的共享任务，请求 A 和 B 都在等同一个 `worker`，A 的页面先关了，不能想当然地把 `worker` 也取消。B 可能还需要结果。反过来，取消 A 自己的等待，也不会自动把共享任务结束；上面这段代码根本没写“最后一位等待者离开时取消 worker”的逻辑。

要支持这种语义，就得另记等待者数量或请求身份，并规定最后一个人离开以后怎么办。这个问题不能靠在 `Task.detached` 前面补个 `[weak self]` 解决：谁拥有任务、谁能取消任务，跟有没有引用环不是一回事。

还有一层旧结果问题。`ChapterScans` 只负责避免同一轮扫描重复执行，不负责决定结果能不能写回当前界面。等待期间用户可能切换阅读位置、移除目录，或者发起更新的一轮扫描。回到 `ReaderModel` 时，仍要核对当前请求身份和目录是否还存在。第三篇讲的“旧任务不能覆盖新选择”，在这里没有因为加了 actor 就自动消失。

现在的个人阅读器还没有接入这个 `ChapterScans`。若真要改，我会先让两条扫描入口都走同一处，再决定定时检查与手动刷新哪些算同一轮；否则新 actor 只是多了一层代码，重复工作还在原地。

actor 解决的是隔离期间谁能改状态。想避免重复加载，还得在挂起前明确登记进行中的工作，并决定哪些请求真的可以分享。`await` 留出的空档，正是这两个问题分开的地方。
