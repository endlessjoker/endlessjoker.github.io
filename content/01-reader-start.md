# 我只是想读个 Markdown，怎么开始写 App 了？

为了读 AI 写的教材，我下载了 Obsidian。用下来总觉得，对我这个需求来说，它有点重。读不同路径下的文件，又得另开窗口。

我只是想要一个阅读器而已！

看来有必要手撸一个了。自己想怎么读，就怎么做。最后还能把项目开源到 GitHub，顺便记录开发中碰到的问题。一鱼多吃，岂不美哉！

这里就用来记这些开发笔记。一个阅读器看着没几项功能，写起来却能牵出不少 SwiftUI 开发中绕不开的问题哦～～

## 先让这些文件待在同一个窗口里

项目叫 `epubBooks`。不过名字起得有点早，目前做的是 **macOS 上的 Markdown 阅读器**，EPUB 还没开始支持。

眼下的需求很具体：教材散落在几个文件夹里，我希望把它们都放在左边的侧栏，点哪篇，右边就显示哪篇。添加另一个目录时，前面的目录也留着。文件继续待在原来的位置，不用为了读一遍，再复制进一个新地方。

现在的代码已经按这个方式组织起来了：SwiftUI 管窗口、侧栏和工具栏，正文交给系统的 WebKit 排版。

工具栏上有一个打开按钮，菜单里也有，按 ⌘O 还能再来一次。三个入口，最后应该操作同一份阅读列表。这个要求太普通了，普通到写界面时很容易直接跳过去。

可往代码里一落，就得先决定：这份数据放哪儿？

## 菜单和窗口，得找同一个 ReaderModel

入口里和打开有关的是这几行：

```swift
import SwiftUI

@main
struct epubBooksApp: App {
    @State private var model = ReaderModel()

    var body: some Scene {
        Window("epubBooks", id: "reader") {
            ContentView(model: model)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("打开文件或文件夹…", action: model.choose)
                    .keyboardShortcut("o")
            }
        }
    }
}
```

这里的 `Window` 声明阅读窗口。添加文件夹之后，变的是窗口里的阅读列表，不用跟着再开一个窗口。

`model` 在 App 这一层创建，然后传给 `ContentView`；菜单里的打开命令也调用它的 `choose()`。

界面里的打开按钮，同样调用 `model.choose`。这个方法会弹出文件选择面板，选好后交给模型的 `open(_:)`，再添加阅读位置、读取目录。

所以点击按钮也好，按 ⌘O 也好，最后找到的是同一个对象。

这里要是顺手又创建一个模型：

```swift
ContentView(model: ReaderModel())
```

类型没错，编译器不会说什么。但窗口拿到了一个新模型，菜单还在用 App 里的那个。

于是菜单忙着往自己的阅读列表里加东西，窗口看着另一份列表，一脸无事发生。遇到这种情况，可能还会怀疑是不是 SwiftUI 没刷新。其实它根本没在看刚才改的对象。

项目现在传的是 `model`，没有重新写一遍 `ReaderModel()`。这点区别很小，却决定了几个入口是不是在做同一件事。

## 等等，class 也能放进 State？

入口这行用的是 `@State`：

```swift
@State private var model = ReaderModel()
```

`ObservableObject` 那套写法里，创建模型常用 `@StateObject`。这里的 `ReaderModel` 换成了 `@Observable`，当前文件、正文、筛选词和阅读进度都放在里面：

```swift
@MainActor
@Observable
final class ReaderModel {
    var selected: URL?
    var markdown = ""
    var filter = ""
    var progress = 0.0
}
```

`@Observable` 让 SwiftUI 能跟踪界面读取了这个对象的哪些属性。比如某处显示 `model.progress`，这处界面就会依赖阅读进度；进度变了，SwiftUI 才知道相关显示需要更新。

而入口的 `@State`，是把这份模型的状态存储交给 SwiftUI 管理。后面需要它的地方，拿到的是这份受管理的引用。这里说的“保存”发生在应用运行期间，不是自动写到磁盘里。

如果把 `@Observable` 去掉，只留下一个普通 class，再往前面加 `@State`，它也不会突然开始追踪所有内部属性的变化。单换一个属性包装器还不够。

`@MainActor` 则管这份界面模型的隔离。它不负责观察变化，也不会自动把读文件这样的工作搬到后台。

## 筛选框为什么又冒出了一个 Bindable

`ContentView` 的筛选框这几行，又用到了 `@Bindable`：

```swift
struct ContentView: View {
    @Bindable var model: ReaderModel

    var body: some View {
        TextField("筛选文件", text: $model.filter)
    }
}
```

`TextField` 得做两件事：显示当前筛选词，以及把新输入的字写回去。如果只交给它一个字符串，它拿到的只是当时的值，还不知道之后应该往哪里写。

`$model.filter` 提供的就是这条读写通道，也就是 `Binding<String>`。有了 `@Bindable`，就能从模型的属性上拿到这个绑定。

所以，在框里输入几个字，改的就是原来那份模型的 `filter`。侧栏生成文件列表时也读取这个属性，筛选框和列表就对上了。这里不需要再保存一份“文本框自己的筛选词”，然后想办法通知列表。

只是显示进度，就用不着绑定了。把这部分单独拎出来的话，可以直接接收引用：

```swift
struct ProgressLabel: View {
    let model: ReaderModel

    var body: some View {
        Text("\(Int(model.progress * 100))%")
    }
}
```

`ProgressLabel` 在 `body` 里读取了可观察的 `progress`，就能建立相应依赖。没有控件索要 `$model.progress`，也就不需要再套一层 `@Bindable`。

## 也不用把整个界面都塞进模型

项目的 `ContentView` 里还留着这样的状态：

```swift
@State private var showOutline = false
```

它只负责章节大纲弹窗现在有没有打开。菜单不需要知道，读文件的逻辑也不需要知道，让这个界面自己管就挺好。

这和 `ReaderModel` 放在 App 那一层并不矛盾。决定放哪儿时，先看看谁要用它，比看见 `@State` 就往同一个地方搬省事。

同样，现在把主阅读模型放在 App 里，是因为这个窗口和菜单要共用它。以后真要做几份互不影响的阅读窗口，就得重新考虑每个窗口各自持有什么。不能因为第一版这么写了，以后所有数据都往这里堆。

再往下就是文件授权了。今天选中的文件夹，明天重新启动 App，还能直接读吗？我原本只是想记住一个路径，macOS 的文件权限已经在后面等着了。
