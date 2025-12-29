# **Orion-Lite 行人重识别（Re-ID）期末项目全流程实现蓝图与深度研究报告**

## **1\. 项目综述与背景分析**

### **1.1 行人重识别（Person Re-ID）的技术图景与挑战**

行人重识别（Person Re-identification，简称 Re-ID）被视为计算机视觉领域中图像检索的一个子问题，其核心任务是在跨摄像头的视频监控网络中，给定一个特定的行人图像（Query），在其他摄像头拍摄的图像库（Gallery）中检索出该行人的所有图像。这项技术在公共安全、智能安防、失踪人口寻找以及商业客流分析中具有极高的应用价值。然而，Re-ID 面临着极具挑战性的技术难点，这主要源于监控场景的复杂性：不同摄像头下的视角变化（Viewpoint Variation）、光照剧烈差异（Illumination Change）、物体遮挡（Occlusion）、低分辨率模糊（Low Resolution）以及行人姿态的多样性（Pose Variation）1。  
传统的 Re-ID 方法主要依赖手工设计的特征（Hand-crafted Features），如颜色直方图、LBP纹理特征等，结合度量学习算法进行匹配。然而，随着深度学习（Deep Learning）的爆发，基于卷积神经网络（CNN）的方法已成为绝对主流。现代 Re-ID 系统通常被设计为一个特征提取与度量学习的端到端框架，其目标是将输入图像映射到一个高维特征空间，使得同一身份（ID）的特征向量距离尽可能小，而不同身份的距离尽可能大。尽管学术界在标准数据集上已取得了超越人类的识别精度（Rank-1 Accuracy \> 95%），但这些高性能模型往往伴随着巨大的参数量（Parameters）和浮点运算量（FLOPs），例如基于 Transformer 的 TransReID 或多分支的 MGN 模型，这使得它们难以在边缘计算设备（如智能摄像头、移动终端或嵌入式开发板）上实时部署。

### **1.2 项目目标：‘Orion-Lite’ 的边缘计算愿景**

本项目名为 **‘Orion-Lite’**，旨在弥合学术界高性能模型与工业界边缘部署需求之间的鸿沟。项目的核心目标是设计并实现一套全流程的 Re-ID 蓝图，专为 **Apple M1 Pro** 芯片架构进行优化。之所以选择 M1 Pro 作为目标硬件，是因为其代表了当前高性能移动端处理器的发展方向——统一内存架构（Unified Memory Architecture, UMA）与专用神经网络加速单元（Apple Neural Engine, ANE）的深度融合。  
‘Orion-Lite’ 不仅仅是一个模型名称，更是一套完整的轻量化解决方案。它要求我们在保持具有竞争力的识别精度（目标 Rank-1 \> 88%）的同时，显著降低模型的计算延迟（目标 Latency \< 10ms/img），实现从数据准备、模型训练、轻量化设计到端侧部署的闭环。本报告将详尽阐述这一蓝图的每一个技术细节，从 Market-1501 数据集的底层结构剖析，到主流重型模型的横向对比，最终推导出 Orion-Lite 的架构设计——一种融合了 **GhostNetV2** 高效骨干与 **Coordinate Attention** 坐标注意力机制的创新架构，并详细说明如何利用 **FastReID** 框架进行训练以及通过 **CoreML** 进行硬件加速部署 3。

## ---

**2\. 数据集深度调研：Market-1501 的解构与分析**

### **2.1 数据集的历史地位与采集方法论**

Market-1501 数据集由清华大学郑良（Liang Zheng）等人于 2015 年在 ICCV 会议上发布，是行人重识别领域最为经典、使用最广泛的基准数据集（Benchmark）之一 5。在 Market-1501 出现之前，Re-ID 领域的数据集（如 VIPeR、CUHK01）普遍规模较小，且图像多为手工裁剪，背景较为单一，难以反映真实监控场景的复杂性。Market-1501 的采集地点位于清华大学校园内的一个繁忙超市门口，涵盖了 6 个摄像头的视角（包括 5 个高分辨率摄像头和 1 个低分辨率摄像头），这为算法提供了丰富的多视角数据和分辨率变化挑战 6。  
该数据集的一个显著特征是其图像框（Bounding Box）是通过 **DPM（Deformable Part Model）** 检测器自动生成的，而非完全人工标注。这意味着数据集中包含了检测误差带来的噪声，例如检测框未对齐、包含过多背景、甚至将非行人目标误检为行人（Distractors）。这种“半自动”标注方式极大地提高了数据集的规模，同时也逼迫算法必须具备鲁棒性，能够处理检测器带来的偏差（Misalignment）。此外，Market-1501 是首个支持大规模 mAP（mean Average Precision）评估的数据集，推动了 Re-ID 评估指标从单一的 CMC 曲线向更加综合的 mAP 转变 7。

### **2.2 目录结构与文件命名协议的严格解析**

为了在 Orion-Lite 项目中正确构建数据加载器（Data Loader），深入理解 Market-1501 的物理存储结构至关重要。根据官方发布的压缩包结构 8，数据集解压后通常包含以下核心目录，每一个都承载着特定的训练或评估逻辑：

* **bounding\_box\_train/（训练集）**：包含 12,936 张 JPEG 图像，涉及 751 个不同的行人身份（Identities）。这些图像用于模型的特征学习阶段。  
* **bounding\_box\_test/（测试集/图库 Gallery）**：包含 19,732 张图像，涉及 750 个行人身份。值得注意的是，训练集和测试集的身份是完全互斥的（Disjoint），这意味着模型必须学习到通用的行人特征描述能力，而非记忆特定的几个人，这被称为“开放集合”（Open-set）识别问题。  
* **query/（查询集/探针 Probe）**：包含 3,368 张图像，是从 750 个测试身份中抽取的图像。在评估时，我们拿一张 Query 图像去 bounding\_box\_test 中搜索属于同一身份的图像。  
* **gt\_query/ & gt\_bbox/**：包含手工标注的 Ground Truth（Matlab 格式）和手工绘制的边界框。这些通常用于验证 DPM 检测器的性能或进行全手工标注的对比实验，但在标准的 Re-ID 深度学习训练流程中较少直接使用 10。

文件命名规则的解码：  
文件名承载了关键的元数据，格式为 0001\_c1s1\_001051\_00.jpg。

* 0001：**Person ID**（身份标签）。范围从 \-1 到 1501。其中 \-1 和 0 通常表示干扰项（Junk/Distractors）或背景，不参与训练分类。  
* c1：**Camera ID**（摄像头编号）。范围 c1 至 c6。Orion-Lite 将利用此信息进行基于摄像头的采样策略或在 TransReID 中用于生成摄像头嵌入（Camera Embedding）。  
* s1：**Sequence ID**（视频序列编号）。表示该图像所属的视频片段。  
* 001051：**Frame ID**（帧号）。表示在视频序列中的时间戳信息，这对于基于视频的 Re-ID 至关重要，但在本项目的单帧图像 Re-ID 中主要作为唯一标识符。  
* 00：**Bounding Box ID**。如果在同一帧中检测到多个边界框，此编号用于区分。

### **2.3 评估协议与核心指标：CMC 与 mAP**

Orion-Lite 项目的成败将通过两个核心指标来量化：**Rank-1 Accuracy** 和 **mAP** 11。理解这两个指标的计算逻辑对于后续的模型调优至关重要。

#### **2.3.1 累积匹配特性（CMC）与 Rank-k**

CMC（Cumulative Match Characteristic）曲线评估的是“检索结果列表中是否存在正确匹配”。

* **Rank-1**：最直观的指标。它表示查询图像的**最相似**匹配项（距离最近的 Gallery 图像）恰好是正确身份的概率。在安防场景中，操作员通常只看第一个结果，因此 Rank-1 权重最高。  
* **Rank-5 / Rank-10**：表示正确匹配出现在前 5 或前 10 个结果中的概率。这在半辅助系统中很有用，操作员可以快速浏览前几张图确认目标。

#### **2.3.2 平均精度均值（mAP）**

当一个查询身份在 Gallery 中有多张图像时（Market-1501 中平均每个 ID 有多张图像），仅仅由 Rank-1 无法反映检索的召回质量。mAP（mean Average Precision）综合考虑了精确率（Precision）和召回率（Recall）。  
计算过程如下：对于每一个 Query，首先计算其平均精度（Average Precision, AP）。AP 是 Precision-Recall 曲线下的面积。简单来说，如果系统将该行人的所有正确图像都排在 Gallery 列表的最前面，AP 就接近 100%。mAP 则是所有 Query 图像 AP 的平均值。一个高 mAP 的模型不仅能找到最像的那张图（高 Rank-1），还能把该行人的所有侧面、背面、遮挡图都找出来并排在前面 12。  
**Junk 图片的处理逻辑**：在 Market-1501 的评估协议中，存在两类“垃圾”（Junk）图像：

1. **同摄像头下的图像**：如果 Query 是摄像头 A 拍的行人 X，那么 Gallery 中摄像头 A 拍的行人 X 的图像被标记为 Junk。这是因为在同一摄像头下追踪行人属于“目标追踪”而非“重识别”，且往往视觉特征极度相似，无法体现跨摄像头的匹配能力。  
2. 背景干扰：检测器误检的纯背景图。  
   在计算 CMC 和 mAP 时，遇到 Junk 图像会直接跳过（既不计入分子也不计入分母），即视而不见，不影响精度的计算 10。

## ---

**3\. 主流模型横向对比与技术剖析**

在构建 Orion-Lite 之前，我们必须对当前学术界的 SOTA（State-of-the-Art）模型进行深入的横向剖析。这不仅是为了设立性能基准（Baseline），更是为了汲取各家之长——BoT 的工程技巧、MGN 的细粒度特征提取、TransReID 的全局注意力机制，并将这些思想融入到我们的轻量化设计中。

### **3.1 BoT (Bag of Tricks): 深度学习工程学的集大成者**

**核心文献**：Luo et al., "Bag of Tricks and a Strong Baseline for Deep Person Re-Identification" (CVPR 2019\) 14。  
BoT 并非一种全新的神经网络架构，而是一套针对 Re-ID 任务特性的训练策略和微调技巧的集合。它的出现打破了“模型越复杂越好”的迷思，证明了通过合理的工程优化，标准的 ResNet-50 骨干网络也能达到惊人的性能（Market-1501 Rank-1 \~94.5%）13。

#### **3.1.1 关键技术细节**

* **Last Stride \= 1**：标准的 ResNet-50 在最后一个残差块（conv5\_x）的下采样步长为 2，导致输出特征图尺寸为 7x7。BoT 将其修改为 1，使得输出尺寸变为 14x14。这一改变显著增加了特征图的空间分辨率，保留了更多有利于区分相似行人的细微细节（如鞋子颜色、衣领形状），尽管这稍微增加了计算量，但收益巨大。  
* **BNNeck (Batch Normalization Neck)**：这是 BoT 最具影响力的创新。在全连接分类层之前加入一个 BN 层。训练时，**ID Loss（交叉熵损失）** 作用于 BN 之后的归一化特征，迫使特征分布在超球面上，有利于分类；而 **Triplet Loss（三元组损失）** 作用于 BN 之前的特征，保留了特征的幅度信息和欧氏距离结构。这种解耦设计解决了两种损失函数在特征空间约束上的潜在冲突 14。  
* **Warmup Learning Rate**：采用“热身”策略，学习率在如前 10 个 Epoch 从 0 线性增加到目标值（如 $3.5 \\times 10^{-4}$）。这避免了在模型权重随机初始化阶段，过大的梯度导致模型陷入局部极小值或震荡，随后在第 40 和 70 Epoch 进行阶梯式衰减 14。  
* **Label Smoothing（标签平滑）**：将 One-hot 标签（如 \`\`）修改为 \[0.1, 0.9, 0.0\]。这防止了模型对训练数据过拟合（Over-confidence），提高了在测试集上的泛化能力。  
* **Random Erasing Augmentation (REA)**：在训练图像中随机选取矩形区域填充随机像素值。这模拟了现实中的遮挡情况（如行人被路灯杆遮挡），强迫网络学习利用局部特征（如只看头部或腿部）进行识别 14。

### **3.2 MGN (Multiple Granularity Network): 极致的切片特征**

**核心文献**：Wang et al., "Learning Discriminative Features with Multiple Granularity for Person Re-Identification" (ACM MM 2018\) 15。  
MGN 的核心哲学是“分而治之”。它认为全局特征（Global Feature）往往忽略了局部细节，因此它设计了一个多分支网络，将图像特征图在垂直方向上进行物理切片（Striping）。

#### **3.2.1 架构解析**

MGN 基于 ResNet-50，但在 res\_conv4\_1 之后将网络分叉为三个独立的分支：

1. **Global Branch**：保留标准的全局平均池化，提取整图特征。  
2. **Part-2 Branch**：将特征图水平切分为上、下两半，分别进行池化和损失计算。  
3. **Part-3 Branch**：将特征图水平切分为上、中、下三部分（头部、躯干、腿部）。

这种设计使得 MGN 能够同时学习到“这个人穿红衣服”（全局）和“这个人穿白鞋”（局部 Part-3）的特征。每个切片都有独立的 Softmax 和 Triplet Loss 进行监督。MGN 在 Market-1501 上通常能达到 Rank-1 95.7% 甚至更高的精度 16。  
**缺点**：计算代价极其高昂。由于三个分支共享了部分底层参数但拥有独立的高层参数，且输出特征维度通常高达 2048 x N，导致推理速度慢，显存占用大。对于 M1 Pro 这样的边缘设备，MGN 的 FLOPs 和参数量是不可接受的负担 17。

### **3.3 TransReID: Transformer 的全局视野**

**核心文献**：He et al., "TransReID: Transformer-based Object Re-Identification" (ICCV 2021\) 18。  
TransReID 标志着 Re-ID 进入了 Transformer 时代。它利用 Vision Transformer (ViT) 的自注意力机制（Self-Attention）来捕捉图像中的长距离依赖关系。

#### **3.3.1 核心创新**

* **纯 Transformer 架构**：将图像切分为 16x16 的 Patch，通过线性投影转化为序列输入 ViT。Self-Attention 允许模型关注图像中任意两个 Patch 之间的关系，无论它们距离多远（例如，它可以直接关联头部的帽子和脚下的鞋子，而 CNN 需要堆叠多层卷积才能扩大感受野）。  
* **Jigsaw Patch Module (JPM)**：受拼图游戏启发，JPM 在训练过程中打乱 Patch 的顺序并重新分组。这迫使模型学习具有位移不变性的鲁棒特征，类似于 CNN 的平移不变性，但通过打乱序列实现 21。  
* **Side Information Embedding (SIE)**：这是 TransReID 最具洞察力的设计。它将摄像头的 ID 和视角的朝向作为可学习的 Embedding 向量，直接加到图像 Patch 的特征上。这使得模型能够显式地去除“摄像头风格偏差”（例如，摄像头 A 偏黄，摄像头 B 偏蓝）。模型学会了“如果是摄像头 A 拍的，就减去 A 的风格向量”，从而还原行人的真实外观 19。

**缺点**：尽管 TransReID 在处理遮挡和多视角方面表现出色（Market-1501 Rank-1 \~95.4%），但 ViT 的计算复杂度与序列长度成二次方关系 $O(N^2)$。其庞大的参数量（ViT-Base 约 86M 参数）和高昂的推理延迟使其难以在算力受限的 M1 Pro 上实现实时（\<30ms）推理 23。

### **3.4 模型横向对比总结表**

| 特性维度 | BoT (ResNet50) | MGN (ResNet50-Mod) | TransReID (ViT-Base) | Orion-Lite (Target) |
| :---- | :---- | :---- | :---- | :---- |
| **骨干架构** | CNN (ResNet) | CNN (Multi-Branch) | Transformer (ViT) | CNN (GhostNetV2) |
| **特征粒度** | 全局特征 | 全局 \+ 局部切片 | 全局 \+ Patch关联 | 全局 \+ 坐标注意 |
| **Market-1501 Rank-1** | \~94.5% | \~95.7% | \~95.4% | **\>88.0% (预期)** |
| **参数量 (Params)** | \~25M | \~60M+ | \~86M | **\<6M** |
| **计算量 (FLOPs)** | \~4.0G | \~10G+ | \~17G | **\<0.5G** |
| **M1 Pro 推理延迟** | \~45ms | \>100ms | \>150ms | **\<10ms** |
| **适用场景** | 服务器端通用方案 | 高精度离线分析 | 复杂遮挡/跨域场景 | **边缘端实时追踪** |

## ---

**4\. Orion-Lite 纵向自研与架构设计蓝图**

基于上述横向对比，Orion-Lite 的设计哲学非常明确：**在 BoT 的工程基础上，利用 GhostNetV2 的极致轻量化骨干，结合 Coordinate Attention 来弥补轻量级网络在空间定位信息上的缺失，从而逼近重型模型的精度。**

### **4.1 骨干网络选型：GhostNetV2 的数学原理**

Orion-Lite 摒弃了常用的 MobileNetV3，转而采用华为诺亚方舟实验室提出的 **GhostNetV2**（NeurIPS 2022） 24。

#### **4.1.1 幽灵模块（Ghost Module）**

传统的卷积层会生成大量冗余的特征图（Feature Maps）。GhostNet 指出，这些冗余特征可以通过廉价的线性变换（Linear Transformation）从少量“本征特征”中生成。  
假设标准卷积生成 $n$ 个特征图。GhostNet 只用标准卷积生成 $m$ 个本征特征图（$m \\ll n$），然后对这 $m$ 个图进行廉价的深度卷积（Depthwise Convolution）生成 $s$ 个“幽灵特征图”。最终将本征图和幽灵图拼接，得到 $n$ 个输出。这在保持特征丰富度的同时，大幅削减了 FLOPs。

#### **4.1.2 DFC Attention：硬件友好的长距离注意力**

GhostNetV1 虽然快，但缺乏捕捉长距离空间信息的能力（这也是 TransReID 强于 CNN 的原因）。GhostNetV2 引入了 DFC（Decoupled Fully-Connected）Attention。与 ViT 中复杂的 Self-Attention 不同，DFC 将特征图在水平和垂直方向分别进行全连接聚合。  
数学上，对于特征图 $Z \\in \\mathbb{R}^{H \\times W \\times C}$，它被分解为：

$$A\_h \= \\text{FC}\_h(Z)$$

$$A\_w \= \\text{FC}\_w(Z)$$

这种分解将注意力机制的复杂度从 $O(H^2W^2)$ 降低到 $O(HW)$，使其能够在 M1 Pro 的神经引擎（ANE）上高效运行，同时捕捉到类似 TransReID 的全局上下文信息（Global Context）27。

### **4.2 注意力增强：Coordinate Attention (CA) 的引入**

虽然 GhostNetV2 自带 DFC，但在 Re-ID 任务中，行人的结构信息（头在上，脚在下）至关重要。标准的 SE-Block（Squeeze-and-Excitation）通过全局平均池化将空间信息压缩为一个点，完全丢失了位置信息。  
Orion-Lite 采用 Coordinate Attention (CA)（CVPR 2021）来替代部分瓶颈层中的 SE 模块 29。  
CA 机制详解：  
CA 不进行全局池化，而是分别沿着高度 $H$ 和宽度 $W$ 方向进行池化，生成两个方向感知特征图：

$$z\_c^h(h) \= \\frac{1}{W} \\sum\_{0 \\le i \< W} x\_c(h, i)$$

$$z\_c^w(w) \= \\frac{1}{H} \\sum\_{0 \\le j \< H} x\_c(j, w)$$

这两个向量分别编码了垂直方向（捕捉人体结构）和水平方向（捕捉侧面特征）的精确位置信息。随后它们被拼接、卷积变换，再重新拆分作用回原特征图。这使得 Orion-Lite 能够精准地“关注”到行人的特定部位（如胸前的 Logo），而不仅仅是统计颜色分布。

### **4.3 Orion-Lite 头部（Head）的定制化设计**

为了适配检索任务，我们对 GhostNetV2 的分类头进行了改造，参考 BoT 的 BNNeck 结构但进行了轻量化：

1. **Global Average Pooling (GAP)**：将 GhostNetV2 输出的特征图（通常是 960 或 1280 维）压缩为向量。  
2. **1x1 Conv 降维层**：引入一个 1x1 卷积层将维度强制压缩至 **512 维**。相比于主流的 2048 维，512 维特征向量在检索阶段的存储空间和余弦相似度计算速度上均有 4 倍的优势，这对于边缘设备的内存管理至关重要。  
3. **BNNeck**：加入 Batch Normalization 层。  
4. **双头输出**：  
   * **Inference 阶段**：直接输出 BN 层的特征 $f\_i$。  
   * **Training 阶段**：  
     * 分支 A：使用 BN 前的特征计算 Triplet Loss。  
     * 分支 B：BN 后的特征接一个全连接层（Linear, 512 \-\> 751），计算 Cross Entropy Loss。

## ---

**5\. 工程实现与训练策略**

### **5.1 训练框架选型：FastReID**

我们选用 JDAI-CV 开发的 **FastReID** 库作为工程底座 4。FastReID 是目前最模块化、扩展性最强的 Re-ID 训练框架，支持配置文件（YAML）管理、自动混合精度训练（AMP）和丰富的评估指标。其高度解耦的设计允许我们仅需编写少量的代码即可注册新的 Backbone（GhostNetV2）和 Head，而复用其成熟的数据加载和评估管线。

### **5.2 M1 Pro 上的环境配置与 MPS 加速**

在 macOS 上训练深度学习模型，利用 M1 Pro 的 GPU 性能依赖于 PyTorch 的 **MPS（Metal Performance Shaders）** 后端 33。  
**环境搭建步骤**：

1. **Python 环境**：建议使用 Miniforge 创建原生 ARM64 环境。  
   Bash  
   conda create \-n orion python=3.9  
   conda activate orion

2. **PyTorch 安装**：  
   Bash  
   pip install \--pre torch torchvision torchaudio \--extra-index-url https://download.pytorch.org/whl/nightly/cpu

   注：虽然稳定版已支持 MPS，但 Nightly 版本通常包含针对 Metal 的最新算子优化，能解决一些复杂层（如 5D Tensor 操作）的回退问题 35。  
3. FastReID 安装：  
   从 GitHub 克隆源码并编译安装，注意跳过不支持 ARM64 的 Cython 扩展（如果遇到编译错误），或者确保 Xcode 命令行工具已更新。

MPS 训练代码适配：  
在 FastReID 的配置文件或启动参数中，必须显式指定设备。虽然 FastReID 默认倾向于 CUDA，但我们可以通过设置 MODEL.DEVICE: "mps" 来强制使用 Metal 后端。  
此外，需要设置环境变量 PYTORCH\_ENABLE\_MPS\_FALLBACK=1。这是因为某些特殊的算子（可能是 GhostNetV2 中的某些插值操作）在 MPS 上尚未完全实现，该标志允许 PyTorch 自动回退到 CPU 执行这些特定算子，避免程序崩溃，虽然会轻微牺牲性能，但保证了能跑通 36。

### **5.3 核心代码实现：Coordinate Attention 模块**

以下是基于 PyTorch 的 Coordinate Attention 实现代码，将嵌入到 GhostNetV2 的反向残差块中 29：

Python

import torch  
import torch.nn as nn  
import math  
import torch.nn.functional as F

class h\_sigmoid(nn.Module):  
    def \_\_init\_\_(self, inplace=True):  
        super(h\_sigmoid, self).\_\_init\_\_()  
        self.relu \= nn.ReLU6(inplace=inplace)  
    def forward(self, x):  
        return self.relu(x \+ 3\) / 6

class h\_swish(nn.Module):  
    def \_\_init\_\_(self, inplace=True):  
        super(h\_swish, self).\_\_init\_\_()  
        self.sigmoid \= h\_sigmoid(inplace=inplace)  
    def forward(self, x):  
        return x \* self.sigmoid(x)

class CoordAtt(nn.Module):  
    def \_\_init\_\_(self, inp, oup, reduction=32):  
        super(CoordAtt, self).\_\_init\_\_()  
        \# 自适应池化，分别压缩 H 和 W 方向  
        self.pool\_h \= nn.AdaptiveAvgPool2d((None, 1))  
        self.pool\_w \= nn.AdaptiveAvgPool2d((1, None))  
          
        mip \= max(8, inp // reduction)  
          
        \# 共享的 1x1 卷积，用于通道降维和特征融合  
        self.conv1 \= nn.Conv2d(inp, mip, kernel\_size=1, stride=1, padding=0)  
        self.bn1 \= nn.BatchNorm2d(mip)  
        self.act \= h\_swish()  
          
        \# 分离卷积，恢复维度  
        self.conv\_h \= nn.Conv2d(mip, oup, kernel\_size=1, stride=1, padding=0)  
        self.conv\_w \= nn.Conv2d(mip, oup, kernel\_size=1, stride=1, padding=0)

    def forward(self, x):  
        identity \= x  
        n, c, h, w \= x.size()  
        \# 1\. 坐标池化  
        x\_h \= self.pool\_h(x)  
        x\_w \= self.pool\_w(x).permute(0, 1, 3, 2\)  
          
        \# 2\. 拼接与变换  
        y \= torch.cat(\[x\_h, x\_w\], dim=2)  
        y \= self.conv1(y)  
        y \= self.bn1(y)  
        y \= self.act(y)   
          
        \# 3\. 拆分与注意力生成  
        x\_h, x\_w \= torch.split(y, \[h, w\], dim=2)  
        x\_w \= x\_w.permute(0, 1, 3, 2\)  
          
        a\_h \= self.conv\_h(x\_h).sigmoid()  
        a\_w \= self.conv\_w(x\_w).sigmoid()  
          
        \# 4\. 施加注意力  
        out \= identity \* a\_w \* a\_h  
        return out

### **5.4 训练配置 (YAML) 详解**

为了在 M1 Pro 上实现 1 小时内的高效训练，我们需要精心调整超参数。以下是 projects/OrionLite/configs/market1501\_ghostnetv2.yml 的关键部分 38：

YAML

\_BASE\_: "../Base-bagtricks.yml"

MODEL:  
  BACKBONE:  
    NAME: "build\_ghostnetv2\_backbone"  
    FEAT\_DIM: 512            \# 强制降维到 512  
    PRETRAIN: True           \# 使用 ImageNet 预训练权重加速收敛  
    PRETRAIN\_PATH: "\~/.cache/torch/checkpoints/ghostnetv2\_torch\_imagenet.pth"  
  HEADS:  
    NAME: "BatchNormHead"  
    NORM: "BN"  
    POOL\_LAYER: "GlobalAvgPool"  
    SCALE: 64                \# Softmax 的 Scaling Factor，有助于收敛  
    MARGIN: 0.35             \# Triplet Loss 的 Margin  
  LOSSES:  
    NAME: ("CrossEntropyLoss", "TripletLoss")  
    TRI:  
      HARD\_MINING: True      \# 开启难样本挖掘  
      MARGIN: 0.3

DATASETS:  
  NAMES: ("Market1501",)  
  TESTS: ("Market1501",)

INPUT:  
  SIZE\_TRAIN:      \# 经典 Re-ID 输入尺寸  
  SIZE\_TEST:   
  REA:  
    ENABLED: True            \# 开启随机擦除  
    PROB: 0.5  
  PADDING: 10                \# 随机填充后裁剪

SOLVER:  
  OPT: "Adam"                \# Adam 收敛比 SGD 快，适合快速实验  
  MAX\_EPOCH: 60              \# 针对轻量化模型，60 Epoch 通常足够  
  BASE\_LR: 0.0008            \# 相比 ResNet，轻量模型可以使用稍大的学习率  
  IMS\_PER\_BATCH: 128         \# M1 Pro 16GB 内存足以支撑 128 BatchSize  
  WARMUP\_ITERS: 1000  
  STEPS:             \# 在第 30 和 50 Epoch 衰减学习率  
  GAMMA: 0.1  
  FP16\_ENABLED: False        \# 警告：MPS 上的 AMP 尚不稳定，建议关闭以保证训练不崩 \[4\]

**训练策略分析**：

* **Batch Size 128**：M1 Pro 的统一内存架构允许 GPU 直接访问 16GB 系统内存，这使得我们可以使用较大的 Batch Size，有利于 BN 层的统计稳定性，同时加速训练吞吐量。  
* **Adam 优化器**：相比 SGD，Adam 对学习率不那么敏感，且在训练初期收敛极快，适合“期末项目”这种时间紧迫的场景。  
* **预训练权重**：加载在 ImageNet 上预训练的 GhostNetV2 权重是必须的，否则从零开始训练在小规模的 Market-1501 上很难达到 85% 以上的精度。

## ---

**6\. 部署方案：从 PyTorch 到 CoreML**

完成训练并获得 model\_final.pth 权重文件后，Orion-Lite 的最后一步是将其转换为 Apple 生态的专用格式 **CoreML** (.mlpackage)，以便在 macOS 或 iOS 应用中调用 M1 Pro 的神经网络引擎（ANE）进行硬件加速推理 40。

### **6.1 模型导出与转换流水线**

这是一个两阶段过程：PyTorch \-\> TorchScript \-\> CoreML。  
步骤 1：模型清洗与 Tracing  
由于训练好的模型包含 Loss 计算分支，我们需要提取纯推理部分。在 FastReID 中，我们可以通过设置 model.eval() 并运行一次前向传播来通过 torch.jit.trace 捕获计算图。

Python

import torch  
from fastreid.config import get\_cfg  
from projects.OrionLite.orion\_model import build\_model  
from fastreid.utils.checkpoint import Checkpointer

\# 1\. 加载配置和权重  
cfg \= get\_cfg()  
cfg.merge\_from\_file("projects/OrionLite/configs/market1501\_ghostnetv2.yml")  
model \= build\_model(cfg)  
Checkpointer(model).load("output/orion\_lite/model\_final.pth")  
model.eval()

\# 2\. 创建 Dummy Input  
\# 必须与训练时的输入尺寸严格一致  
example\_input \= torch.rand(1, 3, 256, 128\) 

\# 3\. JIT Tracing  
traced\_script\_module \= torch.jit.trace(model, example\_input)  
traced\_script\_module.save("orion\_lite\_traced.pt")

步骤 2：使用 CoreMLTools 转换  
利用 Apple 官方的 coremltools 库进行转换。这里有一个关键技巧：使用 ct.ImageType 作为输入类型。这告诉 CoreML 输入是图片，它会自动处理像素归一化（Normalization），这样我们在 Swift 代码中只需传入 CVPixelBuffer，无需手动将像素值除以 255 或减去均值，这能显著减少 CPU 预处理的开销 3。

Python

import coremltools as ct

\# 定义预处理参数：与训练时一致  
\# FastReID 默认是 RGB，范围 ，减去均值除以方差  
\# 这里简化为 1/255 归一化，假设训练时也做了对应调整  
scale \= 1/255.0  
bias \=  

model\_from\_torch \= ct.convert(  
    "orion\_lite\_traced.pt",  
    inputs=,  
    outputs=,  
    convert\_to="mlprogram" \# 使用新版 ML Program 格式  
)

\# 添加元数据  
model\_from\_torch.short\_description \= "Orion-Lite Re-ID Feature Extractor"  
model\_from\_torch.author \= "Orion Project Team"  
model\_from\_torch.version \= "1.0"  
model\_from\_torch.save("OrionLite.mlpackage")

### **6.2 性能优化：量化（Quantization）**

为了进一步极致压缩模型体积和提升推理速度，我们可以对 CoreML 模型进行权重量化。M1 Pro 的 ANE 对 **Int8** 和 **Float16** 计算有极强的加速能力。

* **Float16**：CoreML 默认通常是 Float16（因为 GPU/ANE 主要是 FP16 计算）。精度几乎无损。  
* **Int8 量化**：可以将模型体积再减小一半。虽然可能会带来 0.5%\~1.0% 的 Rank-1 精度下降，但对于追求极致速度的 Orion-Lite 来说是值得的。

Python

\# Int8 线性量化示例  
import coremltools.optimize.coreml as cto  
op\_config \= cto.OpLinearQuantizerConfig(mode="linear\_symmetric", weight\_threshold=512)  
config \= cto.OptimizationConfig(global\_config=op\_config)  
compressed\_model \= cto.linear\_quantize\_weights(model\_from\_torch, config=config)  
compressed\_model.save("OrionLite\_Int8.mlpackage")

### **6.3 部署后的应用逻辑 (Swift)**

在最终的应用程序中，逻辑流如下：

1. **视频流捕获**：使用 AVCaptureSession 获取实时帧。  
2. **行人检测**：调用 Apple 视觉框架（Vision Framework）内置的 VNDetectHumanRectanglesRequest（这个 API 在 iOS/macOS 上经过底层优化，极快）裁剪出行人 ROI。  
3. **特征提取**：将裁剪图像送入 OrionLite.mlpackage，得到 512 维向量。  
4. **相似度匹配**：计算该向量与 Gallery 数据库中存储向量的余弦相似度。如果最高相似度 \> 阈值（如 0.6），则判定为同一人。

## ---

**7\. 预期成果与总结**

### **7.1 预期性能指标**

基于 GhostNetV2 的论文数据以及 Coordinate Attention 的增益预估，Orion-Lite 在 Market-1501 上有望达到以下指标：

| 评估指标 | BoT (ResNet50) | Orion-Lite (预期) | 差异分析 |
| :---- | :---- | :---- | :---- |
| **Rank-1 Accuracy** | 94.5% | **90.5% \~ 91.5%** | 虽然有所下降，但 90%+ 足以满足大多数商业应用。 |
| **mAP** | 85.9% | **76.0% \~ 78.0%** | 细节丢失导致召回率略降，但在可接受范围内。 |
| **模型大小 (MB)** | \~100 MB | **\< 15 MB** | **体积减少 85%**，极大节省存储空间。 |
| **M1 Pro 推理耗时** | \~45 ms/img | **\~6-8 ms/img** | **速度提升 5-7 倍**，实现真正的实时处理。 |

### **7.2 结论**

本项目‘Orion-Lite’通过深入调研 Market-1501 数据集特性，理性分析主流重型模型的优劣，最终提出了一套基于 **GhostNetV2 \+ Coordinate Attention** 的轻量化 Re-ID 架构。利用 **FastReID** 强大的工程能力和 **MPS/CoreML** 的硬件加速特性，我们不仅能在 M1 Pro 上完成模型的高效训练，还能将其部署为极低延迟的实时应用。这一蓝图不仅是一个学术性的期末项目，更为边缘侧计算机视觉应用的落地提供了一个具有参考价值的工程范例。

#### **引用的著作**

1. Reidentify People Throughout a Video Sequence Using ReID Network \- MATLAB & Simulink, 访问时间为 十二月 29, 2025， [https://www.mathworks.com/help/deeplearning/ug/recognize-people-throughout-video-sequence.html](https://www.mathworks.com/help/deeplearning/ug/recognize-people-throughout-video-sequence.html)  
2. ReID-DeePNet: A Hybrid Deep Learning System for Person Re-Identification \- MDPI, 访问时间为 十二月 29, 2025， [https://www.mdpi.com/2227-7390/10/19/3530](https://www.mdpi.com/2227-7390/10/19/3530)  
3. Converting Models to Core ML \- Hugging Face, 访问时间为 十二月 29, 2025， [https://huggingface.co/blog/fguzman82/frompytorch-to-coreml](https://huggingface.co/blog/fguzman82/frompytorch-to-coreml)  
4. \[Quick Review\] FastReID: A Pytorch Toolbox for General Instance Re-identification \- Liner, 访问时间为 十二月 29, 2025， [https://liner.com/review/fastreid-pytorch-toolbox-for-general-instance-reidentification](https://liner.com/review/fastreid-pytorch-toolbox-for-general-instance-reidentification)  
5. Datasets — torchreid 1.4.0 documentation \- Kaiyang Zhou, 访问时间为 十二月 29, 2025， [https://kaiyangzhou.github.io/deep-person-reid/datasets.html](https://kaiyangzhou.github.io/deep-person-reid/datasets.html)  
6. ReID\_Market-1501 \- Kaggle, 访问时间为 十二月 29, 2025， [https://www.kaggle.com/datasets/rayiooo/reid\_market-1501](https://www.kaggle.com/datasets/rayiooo/reid_market-1501)  
7. Person images in the Market-1501 data set. | Download Scientific Diagram \- ResearchGate, 访问时间为 十二月 29, 2025， [https://www.researchgate.net/figure/Person-images-in-the-Market-1501-data-set\_fig4\_334981481](https://www.researchgate.net/figure/Person-images-in-the-Market-1501-data-set_fig4_334981481)  
8. Market-1501 \- CVAT Documentation, 访问时间为 十二月 29, 2025， [https://docs.cvat.ai/docs/dataset\_management/formats/format-market1501/](https://docs.cvat.ai/docs/dataset_management/formats/format-market1501/)  
9. market\_1501 \- Kaggle, 访问时间为 十二月 29, 2025， [https://www.kaggle.com/datasets/pengcw1/market-1501](https://www.kaggle.com/datasets/pengcw1/market-1501)  
10. Market-1501 Dataset \- Academic Torrents, 访问时间为 十二月 29, 2025， [https://academictorrents.com/details/3ea1f8ae1d3155addff586a96006d122587663ee](https://academictorrents.com/details/3ea1f8ae1d3155addff586a96006d122587663ee)  
11. DemacianPrince/Market1501Evaluation: Evaluation codes for Market-1501 Person Re-ID Dataset \- GitHub, 访问时间为 十二月 29, 2025， [https://github.com/DemacianPrince/Market1501Evaluation](https://github.com/DemacianPrince/Market1501Evaluation)  
12. About the Evaluation method of the Market 1501 ReID dataset, 访问时间为 十二月 29, 2025， [https://datascience.stackexchange.com/questions/116375/about-the-evaluation-method-of-the-market-1501-reid-dataset](https://datascience.stackexchange.com/questions/116375/about-the-evaluation-method-of-the-market-1501-reid-dataset)  
13. Enhancing Person Re-Identification via Uncertainty Feature Fusion and Wise Distance Aggregation \- arXiv, 访问时间为 十二月 29, 2025， [https://arxiv.org/html/2405.01101v1](https://arxiv.org/html/2405.01101v1)  
14. Bag of Tricks and a Strong Baseline for Deep Person Re-Identification \- CVF Open Access, 访问时间为 十二月 29, 2025， [https://openaccess.thecvf.com/content\_CVPRW\_2019/papers/TRMTMCT/Luo\_Bag\_of\_Tricks\_and\_a\_Strong\_Baseline\_for\_Deep\_Person\_CVPRW\_2019\_paper.pdf](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)  
15. market1501.py \- levyfan/reid-mgn \- GitHub, 访问时间为 十二月 29, 2025， [https://github.com/levyfan/reid-mgn/blob/master/mgn/market1501.py](https://github.com/levyfan/reid-mgn/blob/master/mgn/market1501.py)  
16. arXiv:1804.01438v1 \[cs.CV\] 4 Apr 2018, 访问时间为 十二月 29, 2025， [https://arxiv.org/pdf/1804.01438](https://arxiv.org/pdf/1804.01438)  
17. Network analysis of particles and grains | Journal of Complex Networks \- Oxford Academic, 访问时间为 十二月 29, 2025， [https://academic.oup.com/comnet/article/6/4/485/4959635](https://academic.oup.com/comnet/article/6/4/485/4959635)  
18. TE-TransReID: Towards Efficient Person Re-Identification via Local Feature Embedding and Lightweight Transformer \- MDPI, 访问时间为 十二月 29, 2025， [https://www.mdpi.com/1424-8220/25/17/5461/review\_report](https://www.mdpi.com/1424-8220/25/17/5461/review_report)  
19. TransReID: Transformer-Based Object Re-Identification \- CVF Open Access, 访问时间为 十二月 29, 2025， [https://openaccess.thecvf.com/content/ICCV2021/papers/He\_TransReID\_Transformer-Based\_Object\_Re-Identification\_ICCV\_2021\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/He_TransReID_Transformer-Based_Object_Re-Identification_ICCV_2021_paper.pdf)  
20. \[2102.04378\] TransReID: Transformer-based Object Re-Identification \- arXiv, 访问时间为 十二月 29, 2025， [https://arxiv.org/abs/2102.04378](https://arxiv.org/abs/2102.04378)  
21. The framework of proposed TransReID. Side Information Embedding (light... \- ResearchGate, 访问时间为 十二月 29, 2025， [https://www.researchgate.net/figure/The-framework-of-proposed-TransReID-Side-Information-Embedding-light-blue-encodes\_fig3\_349123321](https://www.researchgate.net/figure/The-framework-of-proposed-TransReID-Side-Information-Embedding-light-blue-encodes_fig3_349123321)  
22. \[Quick Review\] TransReID: Transformer-based Object Re-Identification \- Liner, 访问时间为 十二月 29, 2025， [https://liner.com/review/transreid-transformerbased-object-reidentification](https://liner.com/review/transreid-transformerbased-object-reidentification)  
23. TE-TransReID: Towards Efficient Person Re-Identification via Local Feature Embedding and Lightweight Transformer \- NIH, 访问时间为 十二月 29, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC12431027/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12431027/)  
24. \[2211.12905\] GhostNetV2: Enhance Cheap Operation with Long-Range Attention \- arXiv, 访问时间为 十二月 29, 2025， [https://arxiv.org/abs/2211.12905](https://arxiv.org/abs/2211.12905)  
25. GhostNetV2: Enhance Cheap Operation with Long-Range Attention | Request PDF, 访问时间为 十二月 29, 2025， [https://www.researchgate.net/publication/365699308\_GhostNetV2\_Enhance\_Cheap\_Operation\_with\_Long-Range\_Attention](https://www.researchgate.net/publication/365699308_GhostNetV2_Enhance_Cheap_Operation_with_Long-Range_Attention)  
26. \[2211.12905\] GhostNetV2: Enhance Cheap Operation with Long-Range Attention \- ar5iv, 访问时间为 十二月 29, 2025， [https://ar5iv.labs.arxiv.org/html/2211.12905](https://ar5iv.labs.arxiv.org/html/2211.12905)  
27. GhostNetV2: Enhance Cheap Operation with Long-Range Attention (Supplementary Material), 访问时间为 十二月 29, 2025， [https://proceedings.neurips.cc/paper\_files/paper/2022/file/40b60852a4abdaa696b5a1a78da34635-Supplemental-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/40b60852a4abdaa696b5a1a78da34635-Supplemental-Conference.pdf)  
28. GhostNetV2: Enhance Cheap Operation with Long-Range Attention \- arXiv, 访问时间为 十二月 29, 2025， [https://arxiv.org/pdf/2211.12905](https://arxiv.org/pdf/2211.12905)  
29. Coordinate Attention for Efficient Mobile Network Design \- CVF Open Access, 访问时间为 十二月 29, 2025， [https://openaccess.thecvf.com/content/CVPR2021/papers/Hou\_Coordinate\_Attention\_for\_Efficient\_Mobile\_Network\_Design\_CVPR\_2021\_paper.pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Hou_Coordinate_Attention_for_Efficient_Mobile_Network_Design_CVPR_2021_paper.pdf)  
30. arXiv:2103.02907v1 \[cs.CV\] 4 Mar 2021, 访问时间为 十二月 29, 2025， [https://arxiv.org/pdf/2103.02907](https://arxiv.org/pdf/2103.02907)  
31. JDAI-CV/fast-reid: SOTA Re-identification Methods and Toolbox \- GitHub, 访问时间为 十二月 29, 2025， [https://github.com/JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid)  
32. yangyueren/fast-reid-video \- GitHub, 访问时间为 十二月 29, 2025， [https://github.com/yangyueren/fast-reid-video](https://github.com/yangyueren/fast-reid-video)  
33. Accelerated PyTorch training on Mac \- Metal \- Apple Developer, 访问时间为 十二月 29, 2025， [https://developer.apple.com/metal/pytorch/](https://developer.apple.com/metal/pytorch/)  
34. Introducing Accelerated PyTorch Training on Mac, 访问时间为 十二月 29, 2025， [https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)  
35. How to run Pytorch on Macbook pro (M1) GPU? \- Stack Overflow, 访问时间为 十二月 29, 2025， [https://stackoverflow.com/questions/68820453/how-to-run-pytorch-on-macbook-pro-m1-gpu](https://stackoverflow.com/questions/68820453/how-to-run-pytorch-on-macbook-pro-m1-gpu)  
36. Is MPS/Apple silicon deprecated now? Why? : r/pytorch \- Reddit, 访问时间为 十二月 29, 2025， [https://www.reddit.com/r/pytorch/comments/1lc09q5/is\_mpsapple\_silicon\_deprecated\_now\_why/](https://www.reddit.com/r/pytorch/comments/1lc09q5/is_mpsapple_silicon_deprecated_now_why/)  
37. Coordinate Attention Explained | Paperspace Blog, 访问时间为 十二月 29, 2025， [https://blog.paperspace.com/coordinate-attention/](https://blog.paperspace.com/coordinate-attention/)  
38. fast-reid/GETTING\_STARTED.md at master \- GitHub, 访问时间为 十二月 29, 2025， [https://github.com/JDAI-CV/fast-reid/blob/master/GETTING\_STARTED.md](https://github.com/JDAI-CV/fast-reid/blob/master/GETTING_STARTED.md)  
39. fastreid.config — fastreid 1.0.0 documentation \- fastreid's documentation\!, 访问时间为 十二月 29, 2025， [https://fast-reid.readthedocs.io/en/latest/modules/config.html](https://fast-reid.readthedocs.io/en/latest/modules/config.html)  
40. Conversion Options \- Core ML Tools, 访问时间为 十二月 29, 2025， [https://coremltools.readme.io/v6.3/docs/neural-network-conversion](https://coremltools.readme.io/v6.3/docs/neural-network-conversion)  
41. CoreML Export for YOLO11 Models \- Ultralytics YOLO Docs, 访问时间为 十二月 29, 2025， [https://docs.ultralytics.com/integrations/coreml/](https://docs.ultralytics.com/integrations/coreml/)  
42. Load and Convert Model Workflow — Guide to Core ML Tools \- Apple, 访问时间为 十二月 29, 2025， [https://apple.github.io/coremltools/docs-guides/source/load-and-convert-model.html](https://apple.github.io/coremltools/docs-guides/source/load-and-convert-model.html)