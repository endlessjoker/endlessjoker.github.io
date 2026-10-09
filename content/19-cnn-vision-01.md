# 视觉为什么要压低自己的响应

CNN 与生物视觉 · 第一篇

电影院暗下来，银幕亮起来。接下来两个小时，我们会看到日光、夜色、爆炸和扑向镜头的物体。散场时，有人意犹未尽，有人揉着眼睛，觉得刚才那块巨大的银幕实在太亮。

一个很自然的猜想是：视觉系统会不会主动压低过强的信号，既省点能量，也保护神经元？如果如此，模仿这种机制的神经网络，学到的究竟是识别能力，还是生物硬件的生存办法？

读 AlexNet 时，这个问题会以一个具体名字出现：侧向抑制，lateral inhibition。作者用它解释局部响应归一化的一部分生物学灵感。一个特征的响应，会受到其他特征响应的压制。

但“压低”只是现象。要理解设计取舍，还得问：**压低以后，信息怎样变了，整个系统少花了多少能量，它又维持了什么样的工作状态？**

这三件事可以相关，却不能互相替代。我们从最容易让人产生怀疑的一幅图开始。

## 看错灰色就说明视觉处理没有用吗

下面两块中央方块使用完全相同的 RGB 值。深色背景上的方块通常显得更亮，浅色背景上的则显得更暗。

![两块 RGB 值同为 128 的灰色方块，分别放在深色和浅色背景上](/cnn-vision/cnn-vision-01/gray-context.svg)

*中央方块均为 RGB(128, 128, 128)。网页版可统一背景。此图用于观看演示，没有经过显示器标定。*

如果任务是读出屏幕的像素值，这种知觉差异确实会误导我们。可是离开屏幕，同样的入眼光强，可能来自阴影里的白纸，也可能来自强光下的灰纸。单看一个位置，无法唯一确定物体表面的反射性质。

在一个简化的漫反射模型里，入眼光强同时受照明和表面反射率影响。要判断物体，视觉系统需要借助上下文；借助上下文，也就有了被特定图案误导的可能。改变场景的透明和分层线索，能让相同的图像区域呈现出显著不同的明暗外观，已有心理物理实验展示了这一点。[1](https://www.nature.com/articles/nature03271)

因此，一张错觉图能说明局部数值与外观并不一一对应，却不能独自判定整个处理策略是否有用，也不能证明它由哪种神经回路造成。

我们需要把问题缩小到一个可以计算的版本：假设想保留的是图案，想忽略的是整体对比度，那么压低响应能帮上什么忙？

## 分母可以消掉大家共同经历的变化

先造一个很小的例子。两个特征探测器对某个图案的响应分别为 3 和 4；只把对比度提高一倍，响应变为 6 和 8。这里假定探测器线性工作，先不考虑噪声和饱和。

两组数值不同，相对关系却完全相同。把每个响应除以这组响应的长度，就得到：

```text
(3, 4) / √(3² + 4²) = (0.6, 0.8)
(6, 8) / √(6² + 8²) = (0.6, 0.8)
```

用符号写，假设 xᵢ = c·sᵢ，其中 sᵢ 是图案结构，c > 0 是所有探测器共同经历的尺度变化。一个教学用的归一化是：

```text
yᵢ = xᵢ / √(ε + Σⱼ xⱼ²)
```

ε 是一个正的小常数，防止分母在信号很弱时接近零。当信号足够强、ε 的影响可以忽略时，共同的 c 近似约掉了。

这时，分母有了比“压制强者”更具体的含义：**它用一组响应估计共同的尺度，让输出更突出相对结构。**

Schwartz 与 Simoncelli 从自然图像和声音的统计结构出发，研究过滤波、整流和除法归一化的组合。他们依据自然信号统计优化模型，并发现模型能描述一些感觉神经元的非线性响应特征。这为归一化提供了高效编码的解释：利用输入的统计规律，减少表示中的冗余。[2](https://www.cns.nyu.edu/~tony/vns/readings/schwartz-simoncelli-2001.pdf)

这不是说每一个生物分母都在执行上面的平方根公式。这个算例负责解释一种可能的收益，论文负责检验更具体的模型与神经响应之间的对应关系。

还要看它舍弃了什么。在 ε = 0 的理想算例里，单凭输出已经分不清原来是 (3, 4) 还是 (6, 8)。如果任务变成比较绝对对比度，这个处理就丢掉了答案。ε > 0 时尺度信息未必完全消失，但输出对尺度的敏感度仍会改变。

所以，“更好的表示”必须附带任务。识别图案与测量强度，未必希望保留同样的信息。

## 输出变小以后为什么还能看得更清楚

还有一种收益来自响应范围的限制。

假设探测器最多只能输出 100。输入 1,000 和 1,100 都被截成 100，区别就消失了；如果在截断之前调整增益，使它们落到 50 和 55，输出更小，差别却保留下来了。

这是一个说明饱和问题的算例，不是真实神经元的量程。真正的问题是：系统既要应对强输入，也要保留对细小变化的敏感度。归一化和适应可以参与这种范围调整，但具体收益取决于噪声、响应非线性和输入分布。[3，Box 2](https://www.cns.nyu.edu/heegerlab/content/publications/Carandini-NRN2012.pdf)

这里很容易犯一个推理错误：既然压低有用，那就一直压低。其实把增益调得过小，弱信号的差别也可能落入后续噪声或量化误差之中。设计目标不是输出越小越好，而是把有限的响应范围用在值得区分的输入上。

不同的“压低”方式，后果也不同。比较两个无量纲教学模型：

<div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>R</mi><mtext>sub</mtext></msub><mo>=</mo><mi>max</mi><mo>(</mo><mn>0</mn><mo>,</mo><mi>E</mi><mo>−</mo><mi>S</mi><mo>)</mo></math></div>

<div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>R</mi><mtext>div</mtext></msub><mo>=</mo><mfrac><msup><mi>E</mi><mn>2</mn></msup><mrow><mn>1</mn><mo>+</mo><msup><mi>E</mi><mn>2</mn></msup><mo>+</mo><msup><mi>M</mi><mn>2</mn></msup></mrow></mfrac></math></div>

第一个减去抑制量 S；第二个让另一种刺激 M 进入分母。演示中为方便比较，令 S = M。固定 E = 3，把 M 从 0 增到 3，再增到 6，除法模型的输出依次约为 0.900、0.474、0.196。

<iframe src="/cnn-vision/cnn-vision-01/demo.html" title="视觉为什么要压低自己的响应：交互实验" style="width:100%;height:1100px;border:0" loading="lazy" sandbox="allow-scripts"></iframe>

*拖动滑块时，曲线表示固定当前 M、扫描 E 所得的响应；圆点表示当前输入。两个模型的纵轴尺度不同，数值大小不能用于判断谁更优。*

减法模型超过阈值后线性增长；这个除法模型则逐渐接近 1。只测到一次响应降低，无法区分它们。需要改变刺激强度和组合，观察整组响应关系。

这类区分也有真实实验基础。鲎眼实验曾通过照亮邻近小眼、记录放电并切断侧向连接，检验相邻输入的抑制作用；猕猴 V1 实验则研究了不偏好的方向刺激如何改变细胞对偏好刺激的响应。[4](https://doi.org/10.1085/jgp.39.5.651) [5，图 10](https://www.cns.nyu.edu/~tony/Publications/carandini-heeger-movshon-1997.pdf)

它们分别为具体系统中的响应关系提供证据。不能因为都出现了“抑制”，就把鲎眼、灵长类皮层和计算机中的一个除法当成同一个机制。

## 神经元省下的能量要从整套回路计算

如果活动被压低，能量是不是也省下来了？要回答，先看能量花在哪里。

神经信号依靠离子跨膜流动。恢复离子浓度梯度需要消耗 ATP；维持静息状态、递质释放和回收也有成本。Howarth 等利用解剖和电生理数据建立能耗模型，分别估计这些过程的开销，并根据新的动作电位数据修订了早期预算。[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC3390818/)

这使“少放电可以少花一部分能量”有了物理基础，但还不能直接推出“增加抑制模块一定省能量”。抑制性神经元本身也需要活动，突触传递和维持回路也有成本。

可以先用一笔概念账来约束我们的判断：

```text
净节省 = 被避免的活动成本 − 新增调节回路的成本
```

这不是一个已经估计好参数的生理模型。它提醒我们选择正确的核算范围：只盯住被压低的那个输出，会漏掉为了压低它而发生的活动。

它还指出一个跨到 CNN 时必须重算的问题。在普通稠密张量计算中，把激活从 100 缩到 10，通常不会减少下一层的乘加次数。要减少计算，需要稀疏执行、低精度等具体机制。激活数值小，与实际少执行了多少工作，是两回事。

因此，一个在生物回路中可能改善能效的策略，移植到数字硬件上以后，可能增加计算开销。仿生设计仍然要面对新硬件的成本结构。

## 稳住活动与保护细胞不是同一个实验终点

“防止太兴奋”的直觉，还包含了两个不同的问题：网络会不会持续失控，细胞会不会受伤。

对前一个问题，稳态可塑性提供了具体机制。Turrigiano 等在培养的皮层神经元中发现，阻断 GABA 介导的抑制，会先提高放电率；约 48 小时后，兴奋性微小突触后电流的幅度降低，放电率回到接近对照的水平。[7](https://pubmed.ncbi.nlm.nih.gov/9495341/)

这里发生的是长期活动变化后的补偿性调整。它与一个刺激出现时，另一个刺激立即改变其响应的现象，处在不同时间尺度上。两者都可能让输出变小，但不能因此把它们合并成一个“保护开关”。

细胞损伤又需要另一套指标。在谷氨酸毒性暴露的培养实验中，研究者通过改变细胞外离子环境，区分了早期肿胀和较迟发生的钙依赖性损伤。[8](https://pmc.ncbi.nlm.nih.gov/articles/PMC6568907/) 另一个实验中，GABA 或所测试的若干抗惊厥药物没有阻止谷氨酸引起的神经元死亡，而一种谷氨酸受体拮抗剂降低了毒性。[9](https://www.neurology.org/doi/10.1212/WNL.37.2.319)

这些结果来自特定的细胞培养和毒性暴露条件。它们说明，改变兴奋性与阻断导致损伤的过程，不能简单画等号。它们也不能用来推断电影院里正常观看银幕会造成同样的暴露或损伤。

至此，“保护”这个宽泛的词需要拆开：维持可用的响应范围、让长期活动恢复稳定、避免细胞损伤，分别需要响应曲线、活动随时间的记录和细胞损伤指标来检验。

## AlexNet 究竟借走了哪一部分

带着这些区分，回到 AlexNet 的局部响应归一化，也就是 LRN。把原论文的通道窗口简写为 N(i)，其计算为：

<div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><msub><mi>b</mi><mi>i</mi></msub><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo><mo>=</mo><mfrac><mrow><msub><mi>a</mi><mi>i</mi></msub><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow><msup><mrow><mo>(</mo><mi>k</mi><mo>+</mo><mi>α</mi><munder><mo>∑</mo><mrow><mi>j</mi><mo>∈</mo><mi>𝒩</mi><mo>(</mo><mi>i</mi><mo>)</mo></mrow></munder><msubsup><mi>a</mi><mi>j</mi><mn>2</mn></msubsup><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo><mo>)</mo></mrow><mi>β</mi></msup></mfrac></math></div>

a 是 ReLU 后的激活。原论文使用同一空间位置上的五个相邻通道，包含当前通道，边界处截断；参数为 k = 2、α = 10⁻⁴、β = 0.75。作者明确指出，通道顺序是在训练前任意确定的。[10，§3.3](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

最值得盯住的是没有变化的 (x, y)。求和改变的是通道编号 j。

![LRN 在固定空间位置跨通道聚合响应，而空间邻域操作查看同一通道的不同位置](/cnn-vision/cnn-vision-01/lrn-neighborhood.svg)

*图中的通道编号不代表固定的方向、颜色或物体类别。两种邻域沿不同的轴定义。*

因此，这一步并不是“左边的亮像素压制右边的暗像素”。它让同一位置上的若干特征响应共同决定分母，但没有规定相邻通道必须具有怎样的生物学关系。

它也不等于赢家通吃。假设只有两个相邻通道非零，激活为 100 和 50，且各自窗口都包含二者。那么它们的分母相同：

<div class="equation"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>D</mi><mo>=</mo><msup><mrow><mo>(</mo><mn>2</mn><mo>+</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>4</mn></mrow></msup><mo>(</mo><msup><mn>100</mn><mn>2</mn></msup><mo>+</mo><msup><mn>50</mn><mn>2</mn></msup><mo>)</mo><mo>)</mo></mrow><mn>0.75</mn></msup><mo>=</mo><msup><mn>3.25</mn><mn>0.75</mn></msup><mo>≈</mo><mn>2.421</mn></math></div>

输出约为 41.31 和 20.66，比例仍然是 2∶1。只留下第一个通道时，其输出约为 43.87。第二个通道确实压低了第一个通道，却没有把较弱的响应删除，也没有在这个算例中放大二者的比例。

更微妙的是，LRN 也没有照搬前面的尺度消除模型。如果把固定非零激活组整体乘以 c，当分母中的常数 k 可以忽略时，分子随 c 增长，分母随 c^(2β) 增长。因此：

```text
LRN 输出的尺度 ∝ c^(1 − 2β)
β = 0.75 时，输出的尺度 ∝ c^(−0.5)
```

在这个强激活近似下，整体输入继续变大，输出反而会缩小。它没有精确保留我们前面算例中的尺度不变性。这是从公式得到的性质，不代表真实网络的所有激活都进入了这个区域。

AlexNet 借用了参照其他响应调节当前响应的想法，却选择了自己的邻域、常数和幂次。要理解它，应当检查这些具体选择，不能让“受生物启发”替代计算。

## 工程收益仍然要由目标系统回答

AlexNet 报告 LRN 在其设置中分别降低 top-1、top-5 错误率 1.4 和 1.2 个百分点。随后 VGG 的配置 A 与 A-LRN，在单尺度验证中的错误率分别为 29.6%／10.4% 和 29.7%／10.5%；作者没有在更深的配置中继续使用 LRN，并指出额外的内存与计算成本。[10](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) [11，表 3](https://arxiv.org/pdf/1409.1556)

这组结果支持的是具体的工程取舍。它既不证明神经归一化没有用，也不能证明 CNN 的准确率提升来自与大脑相同的原因。

即使回到生物实验，也要区分一个功能解释与实现它的主要环节。Pitkow 与 Meister 测到视网膜神经节细胞的输出比视觉输入更少相关，但大部分去相关效果来自非线性处理，而非仅靠中心—周边感受野。[12](https://www.nature.com/articles/nn.3064)

一个合理故事可以帮助提出预测；实验还会告诉我们，故事里的主角是否选对了。

## 带着三个问题回到电影院

现在再问“视觉为什么压低响应”，已经不能只回答“为了保护自己”。在不同系统和条件下，响应调节可能帮助保留相对结构、避免饱和、改变能耗，或者维持长期活动稳定。收益需要和丢失的信息、增加的成本一起计算。

判断一种机制时，可以留下三个具体问题：它保留了什么信息；算上全部相关回路，成本怎样变化；我们测到的稳定或健康，究竟是哪一种指标。

电影院里的问题也需要这样拆。银幕大小与座位改变观看几何；3D 要求双眼处理额外的深度关系；亮度影响入眼光和适应状态。它们不能仅凭“神经元会抑制”这一条原理得到统一答案。

尤其不能把细胞培养里的谷氨酸毒性实验，直接接到“巨幕太亮会伤眼”的结论上。观看不适、短期视功能变化与组织损伤，需要各自的证据。

这个系列会逐步补上那些环节。下一篇先问另一个看似已经解释完的问题：一个细胞偏好某个方向的线条，为什么在计算机里会变成一个遍历整张图像、处处共享参数的卷积核？

等我们回到影院时，希望能说清楚的不只是哪种观影方式更舒服，还包括为什么，以及哪些关于损伤的判断仍然不能从观感推出。

## 延伸阅读

1. Anderson & Winawer（2005），[Image segmentation and lightness perception](https://www.nature.com/articles/nature03271)。上下文与场景分层怎样影响明暗外观。
2. Schwartz & Simoncelli（2001），[Natural signal statistics and sensory gain control](https://www.cns.nyu.edu/~tony/vns/readings/schwartz-simoncelli-2001.pdf)。从自然信号统计解释归一化。
3. Carandini & Heeger（2012），[Normalization as a canonical neural computation](https://www.cns.nyu.edu/heegerlab/content/publications/Carandini-NRN2012.pdf)。重点读 Box 2 与机制讨论。
4. Hartline、Wagner & Ratliff（1956），[Inhibition in the Eye of Limulus](https://doi.org/10.1085/jgp.39.5.651)。鲎眼实验中的响应记录与连接干预。
5. Carandini、Heeger & Movshon（1997），[Linearity and Normalization in Simple Cells of the Macaque Primary Visual Cortex](https://www.cns.nyu.edu/~tony/Publications/carandini-heeger-movshon-1997.pdf)。重点读图 10。
6. Howarth、Gleeson & Attwell（2012），[Updated energy budgets for neural computation in the neocortex and cerebellum](https://pmc.ncbi.nlm.nih.gov/articles/PMC3390818/)。按细胞过程计算能耗，注意模型假设。
7. Turrigiano 等（1998），[Activity-dependent scaling of quantal amplitude in neocortical neurons](https://pubmed.ncbi.nlm.nih.gov/9495341/)。长期活动变化后的突触补偿。
8. Choi（1987），[Ionic dependence of glutamate neurotoxicity](https://pmc.ncbi.nlm.nih.gov/articles/PMC6568907/)。通过离子环境区分损伤过程。
9. Koh & Choi（1987），[Effect of anticonvulsant drugs on glutamate neurotoxicity in cortical cell culture](https://www.neurology.org/doi/10.1212/WNL.37.2.319)。不能把抑制与细胞保护直接等同的实验例子。
10. Krizhevsky、Sutskever & Hinton（2012），[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)。LRN 对应 §3.3。
11. Simonyan & Zisserman（ICLR 2015；预印本 2014），[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556)。LRN 对比见表 3。
12. Pitkow & Meister（2012），[Decorrelation and efficient coding by retinal ganglion cells](https://www.nature.com/articles/nn.3064)。区分去相关的功能解释与实际贡献它的环节。
