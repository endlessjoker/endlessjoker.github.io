# 这个模型，到底该活多久？

搜索模型已经能换一份假的读取实现了。接下来把它放进 SwiftUI，最顺手的写法很可能是：需要搜索页面，就当场创建一个。

但模型里现在装着输入、旧结果和正在运行的任务。创建一个新对象，等于这些东西又从头来了一份。

搜索框里的字还没打完，模型先换人了，这就有点忙过头了。

## 放进 body，意味着它可能再创建

沿用第九篇设计的 `SearchModel`，假如页面这样组装：

```swift
struct RecreatedSearchHost: View {
    let sources: [(UUID, URL)]

    var body: some View {
        SearchEditor(
            model: SearchModel(scan: searchOnDisk),
            sources: sources
        )
    }
}
```

每次这个 `body` 重新求值，`SearchModel(...)` 都会构造新对象。新对象的查询词是空字符串，状态从 `.idle` 开始。

这不是说输入一个字，整个页面一定从最顶层全部重算。SwiftUI 会根据依赖安排更新。问题是，只要这个创建模型的表达式再次执行，它就不会因为下面的界面长得一样，自动把旧模型找回来。

给 `SearchEditor` 加一个固定的 `.id` 也不能阻止这次构造。SwiftUI 管理视图身份，普通的类初始化表达式仍然照常执行。

这个例子是接线时可能写出的错误方式，当前阅读器并没有把 `ReaderModel()` 放在这里。

## 当前阅读器把模型留在入口

真实的应用入口里是 `@State private var model = ReaderModel()`，然后把同一份对象交给 `ContentView(model: model)`。菜单的打开、刷新操作也调用这份模型。

所以窗口中的当前文档和菜单操作没有各存一份。`ContentView` 重新形成一个视图值时，接到的仍然可以是原来的引用。

第九篇的小模型也可以采用同样的方式，只是先把保存范围缩到搜索页面：

```swift
struct SearchScreen: View {
    let sources: [(UUID, URL)]
    @State private var model = SearchModel(scan: searchOnDisk)

    var body: some View {
        SearchEditor(model: model, sources: sources)
    }
}
```

`SearchModel` 采用 `@Observable`，这里用 `@State` 保存它。SwiftUI 为这个视图身份管理状态存储，后续重新创建 `SearchScreen` 这个 struct 值时，不是简单拿新的初始值覆盖已有状态。

所以“视图值又创建了一次”和“状态里的模型换了一次”不能画等号。前者是描述界面的日常工作，后者会影响还在进行的搜索。

但 `@State` 也不是永久保存。这个视图身份结束，它关联的状态存储也会结束。再以新身份出现，就要建立新的一份。

## 子页面需要绑定，不必再藏一份模型

搜索编辑界面只接收对象，并把输入框连到查询词：

```swift
struct SearchEditor: View {
    @Bindable var model: SearchModel
    let sources: [(UUID, URL)]

    var body: some View {
        VStack(alignment: .leading) {
            TextField("搜索正文", text: $model.query)
                .onChange(of: model.query) {
                    model.update(sources: sources)
                }

            if let batch = model.state.displayedBatch {
                Text("“\(batch.request.query)”的结果：\(batch.report.results.count) 条")
            }
        }
    }
}
```

`@Bindable` 让 `$model.query` 可以形成绑定，输入框通过它读写对象的属性。对象仍然是外面传进来的那一个，不会因为包了一层 `@Bindable` 就复制出第二份搜索状态。

如果这里只显示文字，不需要 `$` 绑定，普通的 `let model: SearchModel` 也能在 `body` 读取可观察属性时建立更新依赖。`@Bindable` 不是给整个对象打开观察功能的开关。

也没必要为了“保险”，把传入的 model 再塞进子页面自己的 `@State`。这样会引入另一份由子页面身份管理的存储。父页面以后换了输入对象，子页面的初始值却不一定重新生效，很容易又各用各的。

这里说子页面不负责保存生命周期，并不是说 `@Bindable` 会弱引用模型。它照样可能持有对象。区别在于，谁提供一份跨视图更新继续使用的状态存储，谁只是接收这次界面的输入。

## 关闭搜索以后，还想不想留着上次的词

如果模型由 sheet 内部的 `SearchScreen` 保存，我就不该依赖这份状态在关闭 sheet 后一直留着。下一次展示是否要从空白开始，得先有一个明确的想法。

当前阅读器的做法是把全文搜索字段放在入口持有的 `ReaderModel` 中，sheet 只接收它。关闭搜索页面不会顺手清空这几个字段；再次打开时，还会调用搜索方法处理保留下来的词。

如果拆成小模型以后也想保留这种体验，就把它放在展示 sheet 的父页面：

```swift
struct SearchLauncher: View {
    let sources: [(UUID, URL)]
    @State private var isPresented = false
    @State private var search = SearchModel(scan: searchOnDisk)

    var body: some View {
        Button("全文搜索") { isPresented = true }
            .sheet(isPresented: $isPresented) {
                SearchEditor(model: search, sources: sources)
                    .onAppear { search.update(sources: sources) }
            }
    }
}
```

现在 sheet 关闭，父页面的状态还在，就能继续留着查询词和结果。再次展示时重新搜索，也能处理关闭期间文件内容可能已经变化的情况。

这个保留范围仍然只到 `SearchLauncher` 的身份为止。父页面本身被移除重建，状态也可能重新开始。如果希望跨整个阅读窗口继续保留，就把所有者再放到窗口的稳定根页面；如果希望应用里的多个界面共用，才考虑更高的位置。

当前项目是一个 `Window`，不必为了假想的多窗口提前改动。但以后真换成 `WindowGroup`，在 App 里持有同一个引用再传给所有窗口，通常意味着窗口共用状态。若每个窗口要独立搜索，模型就应该由各自的窗口内容根页面保存。

“往上提”不是越高越好。两个窗口不该互相改搜索词的时候，共享反而是麻烦。

## 初始值，不是每次都执行的配置命令

这里按阅读器当前使用的 Xcode 26.3 来看 `@State` 的行为。写下 `@State private var model = SearchModel(...)`，不等于能保证 `SearchModel.init` 在所有视图构造过程中只执行一次。

重新构造视图值时，初始值表达式仍可能产生临时对象；SwiftUI 随后为已有身份接回原来的状态。状态持续存在和初始化表达式只执行一次，是两个不同的保证。

第九篇的 `SearchModel.init` 只保存闭包，实际搜索由 `update` 启动，所以临时创建没有顺手开一份磁盘扫描。

如果把监听文件、启动网络请求之类的工作塞进初始化，即使最后没用上这个临时对象，副作用也可能已经发生。不能靠一句“有 State 了”就放心。

同样，传给初始化的搜索实现也只是初始选择。如果同一个视图身份已经保存了模型，外部以后换了参数，不该期望这份旧模型自动重新初始化、换掉内部的闭包。需要更换时，应明确更新依赖，或者明确建立新的模型和生命周期。

## 身份一变，之前的保留就不算数了

第四篇用 `.id(model.webSession)` 主动重建网页。它适合阅读引擎需要重新启动的情况，因为当时就是想结束旧对象。

搜索模型也是一样。如果把保存模型的页面写成 `.id(UUID())`，每次产生新 ID，就在要求 SwiftUI 认成另一份视图。原先希望保留的输入和结果，自然也跟着失去原来的状态存储。

还有不写 `.id` 的情况：在两个 `if` 分支里分别放一份 `SearchScreen`，切换分支也可能改变结构身份。看上去还是一个搜索框，不代表它就是原来的那个位置。

如果只是改变排列方式，而搜索必须继续，就把模型放在这两个分支共同的稳定父层里，再传给不同的界面。布局可以换，搜索不必重新开张。

## 模型还活着，任务是不是也该活着

到这里能决定输入和结果保存多久，但任务还多了一层关系。

第九篇的 `Task` 会使用模型的属性，运行期间可以强持有模型。界面不再需要它，不代表模型当场释放；把管理任务的属性丢掉，也不会自动取消那份工作。

反过来，模型仍然由父页面保存，也不代表关闭搜索后继续扫描一定有意义。有时候想保留输入和旧结果，同时停掉还没完成的工作，这三个要求并不冲突。

所以 `.onDisappear`、任务取消、清空输入不能不加区分地绑成一件事。先决定哪份工作属于这次展示，哪份数据属于更长的阅读过程，才知道退出时该结束什么。

下一篇就接着这条线：页面已经关了，谁来告诉任务可以下班了。
