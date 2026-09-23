# 我只是想记住上次打开的文件夹

阅读列表能同时放几个目录以后，下一个要求就很顺手了：关掉 App 再打开，东西应该还在。

总不能每天读书之前，先把昨天那几个文件夹重新选一遍。书还没翻，准备工作倒是一点没少。

乍一想，这有什么难的？把路径存下来，下次再拼成 URL：

```swift
defaults.set(url.path, forKey: "lastFolder")

if let path = defaults.string(forKey: "lastFolder") {
    let url = URL(fileURLWithPath: path)
}
```

路径确实记住了。但 `epubBooks` 开着 App Sandbox，教材又放在应用容器外面。这个 URL 能表示文件夹在哪儿，不等于应用下次启动时就有权限读它。

本来想存个字符串就收工，事情果然没有那么便宜。

## 打开面板还做了一件看不见的事

项目里的 `choose()` 用的是 `NSOpenPanel`，去掉提示文字以后就是这些：

```swift
func choose() {
    let panel = NSOpenPanel()
    panel.canChooseFiles = true
    panel.canChooseDirectories = true
    panel.allowsMultipleSelection = false

    if panel.runModal() == .OK, let url = panel.url {
        open(url)
    }
}
```

这里每次选一个位置，选完追加进阅读列表。`allowsMultipleSelection = false` 限制的是这一次面板里能选几个，不是整个阅读器只能留一个目录。

从这个面板选中文件时，系统同时给应用增加了对所选位置的沙盒访问权限。所以“选完马上能读”和“退出后凭路径再去读”，中间少了一次授权，不能直接当成同一回事。

工程里配了用户所选文件的只读权限：`com.apple.security.files.user-selected.read-only`。只读教材，不改教材，刚好够用。为了跨启动恢复访问，还配了 `com.apple.security.files.bookmarks.app-scope`。

以后要是加正文编辑，那是另一笔账，不能拿着现在这份只读授权就开始写文件。

## 得把书签一起存下来

`open(_:)` 里有这样一段：

```swift
let bookmark = try url.bookmarkData(
    options: [.withSecurityScope, .securityScopeAllowOnlyReadAccess],
    includingResourceValuesForKeys: nil,
    relativeTo: nil
)
```

这里的 bookmark 和“读到第几章，夹张书签”还不是一个东西。它是 Foundation 生成的一段 `Data`，以后可以拿它恢复带安全作用域的 URL。

`.withSecurityScope` 把这个用途写明了，`.securityScopeAllowOnlyReadAccess` 则把后续访问也限制成只读。前提仍然是已经拿到了这个位置的访问权，不能随便编个路径就造一张通行证。

项目把它放在 `SavedLibrary` 里：

```swift
struct SavedLibrary: Identifiable, Codable {
    let id: UUID
    var path: String
    var name: String
    var bookmark: Data
    var isFolder: Bool
    var expanded: Bool
}
```

所以 `path` 没有被扔掉。显示位置、判断是不是重复添加，都还要用它；恢复授权则用 `bookmark`。两样东西一起保存，省得把一个字符串想得太万能。

整个阅读列表经过 `JSONEncoder` 编码，放进 `UserDefaults`。这里存的都是位置和状态，教材正文还老老实实待在原处，没有被塞进偏好设置里。

重启时，`restore()` 先读回这些记录，再逐个解析书签：

```swift
var stale = false
let url = try URL(
    resolvingBookmarkData: library.bookmark,
    options: [.withSecurityScope],
    relativeTo: nil,
    bookmarkDataIsStale: &stale
)

if url.startAccessingSecurityScopedResource() {
    scopedURLs[library.id] = url
}
```

解析书签和开始访问还分成了两步。对于这种从保存的数据恢复出来的安全作用域 URL，拿到 URL 后，要调用 `startAccessingSecurityScopedResource()` 开始访问。

函数名长到换行都嫌碍事，但这一步真省不了。

`stale` 也不能直接理解成“文件没了”。它表示书签数据已经陈旧，解析仍然可能拿到 URL。现在的代码会更新记录里的路径和名字；如果书签陈旧，就尝试重新生成一份，替换掉存着的旧数据。

至于文件到底还在不在、还能不能读，后面照样可能失败。硬盘拔了，书签总不能把硬盘变回来。

## 什么时候算“用完了”

`startAccessingSecurityScopedResource()` 成功之后，还得在不再需要时调用 `stopAccessingSecurityScopedResource()`。开始一次，就得有对应的结束，不能反复开着不管。

按平时处理文件的习惯，很容易想到 `defer`。比如在 `open(_:)` 里顺手这样写：

```swift
let granted = url.startAccessingSecurityScopedResource()
defer {
    if granted {
        url.stopAccessingSecurityScopedResource()
    }
}

scanLibrary(id)
```

看着很规整，进来申请，出去释放。可这里的 `scanLibrary` 会创建异步任务，调用返回时，目录未必已经扫完。

于是 `open(_:)` 一返回，这次申请的访问就先结束了，后台任务却还可能在等着读文件。把 `defer` 放在外面的同步函数里，生命周期就错开了。

即使目录已经扫完，后面点击章节还要读正文，正文里的图片也要读。阅读器里的“用完”，没法简单等同于“打开函数返回”。

现在模型用这个字典留住正在使用的授权位置：

```swift
private var scopedURLs: [UUID: URL] = [:]
```

前面恢复书签时，只有 `startAccessingSecurityScopedResource()` 返回 `true`，才把 URL 放进去。这个字典记着哪些访问是需要结束的；单纯把 URL 放进字典本身并不会申请权限。

从阅读列表移除一个位置时，对应地释放：

```swift
scopedURLs.removeValue(forKey: id)?
    .stopAccessingSecurityScopedResource()
```

模型结束时也会清理剩下的访问。重新打开同一个位置时，旧的那份同样要处理，免得点几次打开就多攒几次申请。

位置还留在阅读列表里时，后续加载正文、读取图片可以继续使用这份授权。不过移除位置时，后台任务是不是已经停稳了，还得跟着异步加载代码一起看；调用取消，也不等于读文件的那段代码会立刻停下来。

## 只选一篇 Markdown，旁边的图片不一定归它管

这也是打开面板同时允许选文件和文件夹的原因。教材经常长这样：

```text
notes/
├── chapter-01.md
└── images/
    └── diagram.png
```

Markdown 里写着：

```markdown
![示意图](images/diagram.png)
```

这个相对路径能算出图片在哪儿，但它不会替应用多申请一份访问权。如果只授权了 `chapter-01.md`，旁边的 `images` 目录并不会因此自动变成可以随便读取的地方。

选中整个 `notes` 文件夹就更符合读教材的需求：这个目录及其子目录会进入本次授权范围。路径上的其他权限限制依然可能让某个文件读失败，但至少不再是只拿着一篇正文的授权去找旁边的图片。

项目自己也检查文件有没有越出选中的位置：先解析符号链接，再判断真实路径。单独打开文件时，只认那个文件；打开文件夹时，才允许读取其中的内容。不能因为 Markdown 写了一个 `../`，就一路追出去。

这和图片怎么排版还没关系。连文件都读不到，CSS 再认真也白忙。

## 读不到，也别把记录直接删了

`restore()` 遇到恢复失败的条目，会把它记进 `unavailableLibraries`，阅读列表里的记录还留着。侧栏给这些位置显示一个不可用标记。

外接盘可能只是没插，目录也可能暂时不可访问。应用此刻读不到，不代表我已经不想读了。要是顺手把记录删掉，等硬盘插回来，又得从头添加，前面折腾保存恢复就很尴尬。

而主动“从阅读列表移除”也只删阅读器里的记录，不删磁盘上的教材。只是看本书，顺手把书删了，这个功能就有点热情过头了。
