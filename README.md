# 🌟 Orion-Lite: 边缘端轻量化行人重识别系统

<p align="center">
  <img src="assets/orion_lite_logo.png" width="200" alt="Orion-Lite Logo">
</p>

<p align="center">
  <b>MobileNetV4 + Circle Loss | M1 Pro 优化 | CoreML 部署</b>
</p>

<p align="center">
  <a href="#-项目简介">简介</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-模型架构">架构</a> •
  <a href="#-性能指标">性能</a> •
  <a href="#-部署">部署</a>
</p>

---

## 📖 项目简介

**Orion-Lite** 是一个面向边缘计算设备的轻量化行人重识别（Person Re-ID）系统，专为 **Apple M1 Pro** 芯片优化。项目融合了高效骨干网络与先进的度量学习技术，在保持高精度的同时实现了极低的推理延迟。

### 🎯 核心特性

- **极致轻量**: 模型参数量 < 4M，FLOPs < 0.3G
- **高效推理**: M1 Pro 上推理延迟 < 3ms/img
- **竞争精度**: Market-1501 Rank-1 > 90%
- **端侧部署**: 支持 CoreML 格式，可直接在 iOS/macOS 设备运行
- **训推一体**: 完整支持在 M1 Pro 上完成训练和推理

### 🚀 v2.0 技术选型

| 组件 | 选型 | 论文出处 |
|------|------|----------|
| **骨干网络** | MobileNetV4-Conv-Small | arXiv:2404.10518 (ECCV 2024) |
| **度量学习** | Circle Loss | arXiv:2002.10857 (CVPR 2020) |
| **分类损失** | Cross Entropy + Label Smoothing | BoT (CVPR 2019) |
| **头部设计** | BNNeck | BoT (CVPR 2019) |

---

## 📁 项目结构

```
Orion-Lite/
├── orion_lite/              # 🔧 核心库
│   ├── configs/             # 配置文件
│   ├── data/                # 数据加载与增强
│   ├── engine/              # 训练/评估引擎
│   ├── losses/              # 损失函数 (Circle Loss, CE)
│   ├── metrics/             # 评估指标 (CMC, mAP)
│   ├── modeling/            # 模型定义
│   │   ├── backbones/       # 骨干网络 (MobileNetV4, GhostNetV2)
│   │   ├── heads/           # 头部网络 (BNNeck)
│   │   └── layers/          # 自定义层 (CoordAtt, Pooling)
│   ├── optim/               # 优化器与学习率调度
│   └── utils/               # 工具函数
├── scripts/                 # 📜 主入口脚本
│   ├── train.py             # 训练入口
│   ├── test.py              # 测试入口
│   └── demo.py              # 演示脚本
├── tools/                   # 🛠️ 工具脚本
│   ├── deploy/              # 模型导出与部署
│   └── visualize/           # 可视化工具
├── checkpoints/             # 💾 模型权重
├── logs/                    # 📊 训练日志
├── demo/                    # 🎬 演示资源
├── docs/                    # 📚 文档
└── assets/                  # 🖼️ 静态资源
```

---

## 🚀 快速开始

### 环境配置

```bash
# 激活虚拟环境
conda activate orion_lite

# 验证 MPS 可用性
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

### 训练模型

```bash
python scripts/train.py --config orion_lite/configs/market1501_ghostnetv2.yaml
```

### 评估模型

```bash
python scripts/test.py --config orion_lite/configs/market1501_ghostnetv2.yaml \
                       --checkpoint checkpoints/orion_lite_best.pth
```

### 导出 CoreML

```bash
python tools/deploy/export_coreml.py --checkpoint checkpoints/orion_lite_best.pth \
                                     --output OrionLite.mlpackage
```

---

## 🏗️ 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Input (3×256×128)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              MobileNetV4-Conv-Small Backbone                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Universal Inverted Bottleneck (UIB)                 │   │
│  │  参数量: ~2.5M | Apple ANE 优化                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BNNeck Head                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  GAP → 1×1 Conv (1280→512) → BatchNorm → 双头输出     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     ┌─────────────────┐             ┌─────────────────┐
     │  Circle Loss    │             │    ID Loss      │
     │   (BN 前特征)    │             │  (BN 后特征)     │
     └─────────────────┘             └─────────────────┘
```

---

## 📊 性能指标

| 指标 | Orion-Lite | BoT (ResNet50) | 对比 |
|------|------------|----------------|------|
| **Rank-1** | ~92% | 94.5% | -2.5% |
| **mAP** | ~79% | 85.9% | -6.9% |
| **参数量** | 3.5M | 25M | **-86%** |
| **推理延迟** | ~2.3ms | ~45ms | **-95%** |
| **模型大小** | ~14MB | ~100MB | **-86%** |

### M1 Pro 实测数据

| 骨干网络 | 参数量 | MPS 延迟 | 吞吐量 |
|----------|--------|----------|--------|
| MobileNetV4-Small | 2.49M | 2.26ms/img | ~440 img/s |
| GhostNetV2-100 | 4.88M | 8.50ms/img | ~118 img/s |

---

## 🔧 技术栈

- **训练框架**: PyTorch 2.9+ with MPS Backend
- **骨干网络**: MobileNetV4 (timm)
- **度量损失**: Circle Loss
- **部署格式**: CoreML (.mlpackage)
- **目标硬件**: Apple M1 Pro (ANE 加速)

---

## 📜 许可证

本项目仅供学术研究和教育用途。

---

## 📚 参考文献

### 核心论文

1. **MobileNetV4**: Qin, D., et al. "MobileNetV4 - Universal Models for the Mobile Ecosystem." *ECCV 2024*. [[arXiv:2404.10518](https://arxiv.org/abs/2404.10518)]

2. **Circle Loss**: Sun, Y., et al. "Circle Loss: A Unified Perspective of Pair Similarity Optimization." *CVPR 2020*. [[arXiv:2002.10857](https://arxiv.org/abs/2002.10857)]

3. **GhostNetV2**: Tang, Y., et al. "GhostNetV2: Enhance Cheap Operation with Long-Range Attention." *NeurIPS 2022*. [[arXiv:2211.12905](https://arxiv.org/abs/2211.12905)]

4. **Coordinate Attention**: Hou, Q., et al. "Coordinate Attention for Efficient Mobile Network Design." *CVPR 2021*. [[arXiv:2103.02907](https://arxiv.org/abs/2103.02907)]

5. **BoT (Bag of Tricks)**: Luo, H., et al. "Bag of Tricks and a Strong Baseline for Deep Person Re-Identification." *CVPR 2019 Workshop*. [[CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)]

6. **TransReID**: He, S., et al. "TransReID: Transformer-based Object Re-Identification." *ICCV 2021*. [[arXiv:2102.04378](https://arxiv.org/abs/2102.04378)]

7. **MGN**: Wang, G., et al. "Learning Discriminative Features with Multiple Granularity for Person Re-Identification." *ACM MM 2018*. [[arXiv:1804.01438](https://arxiv.org/abs/1804.01438)]

### 工具与框架

8. **FastReID**: He, L., et al. "FastReID: A Pytorch Toolbox for General Instance Re-identification." [[GitHub](https://github.com/JDAI-CV/fast-reid)]

9. **timm**: Wightman, R. "PyTorch Image Models." [[GitHub](https://github.com/huggingface/pytorch-image-models)]

10. **CoreML Tools**: Apple Inc. "Core ML Tools." [[Apple Developer](https://coremltools.readme.io/)]

---

<p align="center">
  <b>人工智能技术结课项目 | 青岛大学</b>
</p>
