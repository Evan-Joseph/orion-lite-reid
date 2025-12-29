# 🌟 Orion-Lite: 边缘端轻量化行人重识别系统

<p align="center">
  <img src="assets/orion_lite_logo.png" width="200" alt="Orion-Lite Logo">
</p>

<p align="center">
  <b>GhostNetV2 + Coordinate Attention | M1 Pro 优化 | CoreML 部署</b>
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

**Orion-Lite** 是一个面向边缘计算设备的轻量化行人重识别（Person Re-ID）系统，专为 **Apple M1 Pro** 芯片优化。项目融合了 GhostNetV2 的高效骨干网络与 Coordinate Attention 的空间定位能力，在保持高精度的同时实现了极低的推理延迟。

### 🎯 核心特性

- **极致轻量**: 模型参数量 < 6M，FLOPs < 0.5G
- **高效推理**: M1 Pro 上推理延迟 < 10ms/img
- **竞争精度**: Market-1501 Rank-1 > 88%
- **端侧部署**: 支持 CoreML 格式，可直接在 iOS/macOS 设备运行
- **训推一体**: 完整支持在 M1 Pro 上完成训练和推理

## 📁 项目结构

```
Orion-Lite/
├── orion_lite/              # 🔧 核心库
│   ├── configs/             # 配置文件
│   ├── data/                # 数据加载与增强
│   ├── engine/              # 训练/评估引擎
│   ├── losses/              # 损失函数 (Triplet, CE, etc.)
│   ├── metrics/             # 评估指标 (CMC, mAP)
│   ├── modeling/            # 模型定义
│   │   ├── backbones/       # 骨干网络 (GhostNetV2)
│   │   ├── heads/           # 头部网络 (BNNeck)
│   │   └── layers/          # 自定义层 (CoordAtt)
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

## 🏗️ 模型架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Input (3×256×128)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  GhostNetV2 Backbone                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Ghost Module + DFC Attention (长距离依赖)           │   │
│  │  参数量: ~6M | FLOPs: ~0.5G                          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Coordinate Attention                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  水平/垂直方向感知 → 精确定位人体部位                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    BNNeck Head                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  GAP → 1×1 Conv (512-d) → BatchNorm → 双头输出        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     ┌─────────────────┐             ┌─────────────────┐
     │  Triplet Loss   │             │    ID Loss      │
     │   (BN 前特征)    │             │  (BN 后特征)     │
     └─────────────────┘             └─────────────────┘
```

## 📊 性能指标

| 指标 | Orion-Lite | BoT (ResNet50) | 对比 |
|------|------------|----------------|------|
| **Rank-1** | ~90% | 94.5% | -4.5% |
| **mAP** | ~77% | 85.9% | -8.9% |
| **参数量** | 6M | 25M | **-76%** |
| **推理延迟** | ~7ms | ~45ms | **-84%** |
| **模型大小** | ~15MB | ~100MB | **-85%** |

## 🔧 技术栈

- **训练框架**: PyTorch 2.9+ with MPS Backend
- **骨干网络**: GhostNetV2 (timm)
- **注意力机制**: Coordinate Attention
- **部署格式**: CoreML (.mlpackage)
- **目标硬件**: Apple M1 Pro (ANE 加速)

## 📜 许可证

本项目仅供学术研究和教育用途。

## 📚 参考文献

- GhostNetV2: Enhance Cheap Operation with Long-Range Attention (NeurIPS 2022)
- Coordinate Attention for Efficient Mobile Network Design (CVPR 2021)
- Bag of Tricks for Deep Person Re-Identification (CVPR 2019)
- FastReID: A Pytorch Toolbox for General Instance Re-identification

---

<p align="center">
  <b>人工智能技术结课项目 | 青岛大学</b>
</p>
