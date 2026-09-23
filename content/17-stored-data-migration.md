# 保存的数据，下一版还读得回来吗？

阅读器关掉再打开，左边的教材目录还在；书签和笔记也还在。这些东西不像搜索结果，算错了下次再搜就行。既然答应记住，就得考虑下一版代码还能不能认出上一版存的字节。

现在阅读位置存成 `libraries.v1`，笔记存成 `readingNotes.v1`。`SavedLibrary` 和 `ReadingNote` 都遵守 `Codable`，用 `JSONEncoder` 编成数据，再交给 `UserDefaults`。名字里已经有 `v1`，像是提前给未来留了门牌。可门牌写了版本号，不等于迁移已经发生。

## 一行 try?，能藏住多少消息

启动恢复阅读位置时，当前代码大意就是这段：

```swift
if libraries.isEmpty,
   let data = defaults.data(forKey: "libraries.v1"),
   let saved = try? JSONDecoder().decode([SavedLibrary].self, from: data) {
    libraries = saved
}
```

没有数据，就保持空列表；数据损坏或类型对不上，也保持空列表。这两种情况在界面上长得一样，但原因完全不同。更要紧的是，`restore()` 后面还会调用 `persistLibraries()`。如果旧数据解码失败，内存里的 `libraries` 仍是空的，这次保存就可能把原来的 `libraries.v1` 覆盖成空数组。

我没在阅读器里遇到过这种丢失。风险是沿着现有控制流推出来的：解码失败被 `try?` 吞掉，恢复流程却没有因此停下保存。

笔记的入口也用 `try?` 解码 `readingNotes.v1`。它不会在构造模型的同一刻立刻保存空笔记，但如果之后新增或删除笔记，`persistNotes()` 就会按当时内存中的数组写回。这里同样不该把“读不出旧数据”当成“用户本来没有笔记”。

## 新属性有默认值，旧 JSON 也未必买账

假设下一版想给阅读位置加一个“置顶”标记。容易以为写个默认值就结束了：

```swift
struct LibraryV1: Codable {
    var name: String
}

struct LibraryV2: Codable {
    var name: String
    var pinned = false
}
```

旧数据只有 `name`。合成的 `Decodable` 不会因为 `pinned` 在属性声明处写了 `false`，就自动把缺失的 JSON 键补上；按这个例子解码会得到 `keyNotFound`。`Codable` 会替我生成编码和解码代码，但它不知道“没有 pinned 就当没置顶”是我想要的迁移规则。

如果只加这一个兼容字段，可以自己实现 `init(from:)`，用 `decodeIfPresent(Bool.self, forKey: .pinned) ?? false` 明确写出旧值的意义。字段越来越多、结构也要改时，保留一份旧格式类型、显式转成新格式，反而更容易看出究竟改了什么。

## 先读旧版，转好以后再写新版

沿用现有 `SavedLibrary` 的字段，假设新版另加一组置顶位置。新格式可以包一层：

```swift
struct LibraryStoreV2: Codable {
    let libraries: [SavedLibrary]
    let pinnedIDs: Set<UUID>
}

func loadLibraryStore(from defaults: UserDefaults) throws -> LibraryStoreV2? {
    let decoder = JSONDecoder()

    if let data = defaults.data(forKey: "libraries.v2") {
        return try decoder.decode(LibraryStoreV2.self, from: data)
    }

    guard let oldData = defaults.data(forKey: "libraries.v1") else {
        return nil
    }
    let old = try decoder.decode([SavedLibrary].self, from: oldData)
    let migrated = LibraryStoreV2(libraries: old, pinnedIDs: [])
    let newData = try JSONEncoder().encode(migrated)
    defaults.set(newData, forKey: "libraries.v2")
    return migrated
}
```

返回 `nil` 只表示两个版本都没存过。只要存过数据却解码失败，函数就抛出错误，调用处要停下这轮自动保存，留下原始字节和一条可见的恢复提示。不能在 `catch` 里悄悄改成 `[]`，随后照常写回。

迁移成功时，先完整解码旧数组、建好新值、编码成功，才写 `libraries.v2`。旧的 `libraries.v1` 暂时保留。即使新版写入没有按预想持久化，下次仍有旧数据可重试；这里的 `UserDefaults.set` 不是数据库事务，我也不打算把它说成“已原子提交”。什么时候清掉旧键，可以等新版恢复路径被验证后再决定。

如果 `libraries.v2` 已经存在却解码失败，这段代码会报错，不会退回 `v1` 然后把坏掉的 `v2` 覆盖掉。是否提供“从备份恢复”的操作，可以另做明确的选择；自动装作没看见，正是上一节的问题。

这只是一个可读的迁移骨架，尚未写入个人阅读器。真正接到 `restore()` 时，还要让失败状态阻止后面的 `persistLibraries()`，否则迁移函数再谨慎，外面仍会把数据抹掉。

## 解码成功，也不等于文件能打开

`SavedLibrary` 里面还存着安全作用域书签。JSON 能解出 `Data`，只能证明存储格式读得通；书签能不能解析到文件、文件是否还在、访问是否获准，是下一段恢复流程的事。

现有代码会逐个解析书签、尝试开始访问、检查文件是否存在；解析或文件存在性检查失败时，留下记录并标为不可用，不是把整个阅读列表一把清空。书签被标记为过期但还能解析时，它还会尝试生成新书签。这个“书签过期”属于文件访问层的更新，和 `libraries.v1` 变成 `libraries.v2` 不是同一种版本迁移。

项目里还有更老的单一 `libraryBookmark` 键：阅读位置列表为空时，会尝试从它重新打开一个位置。这是已有的兼容路径。不过如果新版列表解码失败，不能拿“还能找回一个旧书签”当理由覆盖整份列表；它们保存的信息量不同。

笔记以后若改格式，也该给 `readingNotes.v1` 留相同的退路。注释、收藏、阅读位置都不是能随便重建的缓存。用户自己写过的字，比我少写几行迁移代码值钱得多。`Codable` 负责把格式翻译成类型；哪些旧数据可以保留、哪些失败必须停下来，还得由我决定。
