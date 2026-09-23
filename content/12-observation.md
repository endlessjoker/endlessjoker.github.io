# 这个属性变了，为什么这里没刷新？

阅读器底部有个百分比。网页滚动后把进度传回来，`ReaderModel.saveProgress` 接住它：

```swift
func saveProgress(_ value: Double, document: String) {
    guard document == selected?.path else { return }
    progress = min(1, max(0, value))
    defaults.set(progress, forKey: (companion ? "comparisonProgress:" : "progress:") + document)
}
```

底部则直接写着 `Text("\(Int(model.progress * 100))%")`。这套代码目前用的是 `@Observable`。要是有一天百分比没动，我大概会先怀疑网页消息没送到，然后才怀疑状态系统。毕竟一眼看过去，“值已经变了，界面怎么还没变”很像 SwiftUI 的玄学。但这里其实有几道不同的门：消息有没有到，`saveProgress` 有没有通过文档路径检查，界面有没有读取这份属性。前两道是数据流，最后一道才是 Observation。

## 被观察的不是“整个对象动了”

`ReaderModel` 声明为 `@MainActor @Observable`，其中 `progress` 是普通的存储属性。Swift 编译器展开 `@Observable` 后，会给属性的读取和修改接上观察登记。视图构造内容时读了 `model.progress`，这次读取就能成为更新依赖。以后它变化，SwiftUI 才有理由重新检查相关视图。

同一个模型里还有 `libraries`。侧栏底部显示 `model.libraries.count`。从数据语义看，翻一页书不会增减阅读位置，进度和位置数是两件事。Observation 的属性跟踪也允许它们分别成为依赖，不必因为都住在 `ReaderModel` 里，就把每个属性都当成同一个开关。

不过当前 `ContentView` 很大：侧栏、阅读区、底栏都在其中，代码里也同时读取了不少模型属性。不能看见“按属性跟踪”，就宣布滚动时侧栏一定连求值都不会发生。观察粒度说的是依赖怎么登记；视图如何拆分、SwiftUI 具体评估哪些内容，是下一层问题。要让进度显示的依赖更清楚，可以把它提出来：

```swift
struct ProgressBadge: View {
    let model: ReaderModel

    var body: some View {
        Text("\(Int(model.progress * 100))%")
            .monospacedDigit()
    }
}
```

这里只读进度，不需要写回模型，`let model` 就够了。第十篇用 `@Bindable` 是因为输入框要 `$model.query`；它不是“想让文字刷新就必须加”的咒语。这个 `ProgressBadge` 只是我在纸面上拆出的版本，当前阅读器底部仍直接写在 `ContentView` 里。

## 看起来读了模型，实际只读了一次

如果真遇到进度数字停住，下面这种写法比“SwiftUI 坏了”更值得查：

```swift
struct FrozenProgress: View {
    let model: ReaderModel
    @State private var shown = 0

    var body: some View {
        Text("\(shown)%")
            .onAppear {
                shown = Int(model.progress * 100)
            }
    }
}
```

`body` 用来画字的值是 `shown`。`model.progress` 只在出现回调里被取了一次，复制进另一份状态。后来进度再变，复制品不会自动跟着变。即使别的原因让界面重算，`shown` 仍是旧数字。

这是一个假设的错误版本，阅读器没有这样保存百分比。它提醒我区分“读过一次”与“视图构造内容时依赖它”。把模型属性复制进 `@State`，等于自己接管同步；如果只是显示，根本没必要做这次复制。

有时想给进度取个更合适的名字，计算属性也可以：

```swift
var progressPercent: Int {
    Int(progress * 100)
}
```

放在模型里，视图读取 `model.progressPercent` 时，计算过程会读到被跟踪的 `progress`，依赖照样能建立。这里的关键不是“视图源码里必须出现 progress 这个单词”，而是这次求值最终有没有走到可观察属性的读取。倘若计算值来自另一个不会经过这些访问器的系统，给外面套一层计算属性并不能凭空产生通知。

## 换回 ObservableObject，会发生什么

为了弄明白这点，我把老写法缩成只剩两个字段：

```swift
@MainActor final class LegacyReadingModel: ObservableObject {
    @Published var progress = 0.0
    @Published var markdown = ""
}

struct LegacyProgressBadge: View {
    @ObservedObject var model: LegacyReadingModel

    var body: some View {
        Text("\(Int(model.progress * 100))%")
    }
}
```

`ObservableObject` 这一路用 Combine 的变化发布机制。`@Published` 的属性改动会通过对象的 `objectWillChange` 让订阅它的视图知道：这个对象要变了。上面的视图虽只显示 `progress`，`markdown` 改动也能让这份对象订阅收到变化；它不会先检查 `LegacyProgressBadge.body` 究竟读过哪个属性。至于 SwiftUI 之后会怎么比对和更新界面，不能仅凭收到通知推断成“整棵 UI 都重画了”。

而 `@Observable` 这一路，编译器把普通存储属性改造成带访问记录的属性，并由 `ObservationRegistrar` 管理访问和变化。它关心的是某次视图求值具体读了什么。前者更像对象发出一声“我要变了”，后者还记得“你上次问的是哪一项”。这句只是帮助我记机制，不是承诺某种性能数字。

我用当前 Swift 编译器展开了一个只有 `progress` 和 `markdown` 的小类。`progress` 背后出现了私有存储 `_progress`；读取时走 `access(keyPath: \.progress)`，修改时走 `withMutation(keyPath: \.progress)`。所以源码里看见“普通 var”不代表运行时只是裸字段。宏把登记工作补到了访问器里。

老模型如果有个字段既没 `@Published`，也没自己发送 `objectWillChange`，改它不能指望这套订阅自动知道。新模型则不需要给每个普通存储属性单独写 `@Published`。迁移时真正要核对的，是视图读到了哪些属性，以及原先依赖对象整体通知的地方是否偷偷借它刷新了别的内容。

## Observation 也不负责安排线程

`@Observable` 没有把模型变成 actor。阅读器另外写了 `@MainActor`，所以 `progress` 的访问受主 actor 隔离约束。后台算完进度，不能因为属性“可观察”就随便从后台直接改它；该回主 actor 的地方还是要回。

反过来，所有修改都回到了主 actor，也不保证数字就会变。`saveProgress` 可能因为当前文档路径不匹配而直接返回；或者界面显示的是像 `shown` 那样单独保存的旧值。线程安全、状态有没有写入、视图有没有依赖这份状态，是三笔不同的账。把它们混成一句“SwiftUI 没刷新”，查起来只会越来越像猜谜。
