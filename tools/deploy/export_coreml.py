# -*- coding: utf-8 -*-
"""
CoreML 模型导出工具

将训练好的 PyTorch 模型导出为 Apple CoreML (.mlpackage) 格式，
以便在 iOS/macOS 设备上部署。

功能：
1. 加载训练好的权重
2. 转换为 TorchScript (Trace)
3. 使用 coremltools 转换为 CoreML
4. 验证导出模型的一致性
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import coremltools as ct
from PIL import Image
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from orion_lite.modeling import OrionLite


def parse_args():
    parser = argparse.ArgumentParser(description='Orion-Lite CoreML Export')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重路径 (.pth)')
    parser.add_argument('--output', type=str, default='OrionLite.mlpackage',
                        help='导出模型路径')
    parser.add_argument('--backbone', type=str, 
                        default='mobilenetv4_conv_small',
                        help='骨干网络名称')
    parser.add_argument('--num_classes', type=int, default=751,
                        help='类别数 (需与训练时一致)')
    parser.add_argument('--height', type=int, default=256, help='输入高度')
    parser.add_argument('--width', type=int, default=128, help='输入宽度')
    
    return parser.parse_args()


class ExportModel(nn.Module):
    """
    导出专用封装模型
    
    仅包含推理所需的部分：骨干网络 + BNNeck (推理模式)。
    移除训练时的分类头等。
    """
    def __init__(self, original_model):
        super(ExportModel, self).__init__()
        self.backbone = original_model.backbone
        self.head = original_model.head
        
        # 确保处于评估模式
        self.eval()
        
    def forward(self, x):
        # 1. 骨干提取特征
        features = self.backbone(x)
        
        # 2. 头部处理 (Global Pooling + BN)
        # 注意：BNNeckHead 在 eval 模式下只返回归一化后的特征
        bn_features = self.head(features)
        
        return bn_features


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("🌟 Orion-Lite CoreML Exporter")
    print("=" * 60)
    
    # 1. 加载 PyTorch 模型
    print(f"\n📦 Loading model from {args.checkpoint}...")
    model = OrionLite(
        backbone_name=args.backbone,
        num_classes=args.num_classes,
        pretrained=False, # 权重将从 checkpoint 加载
    )
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # 处理 state_dict 键名可能存在的 'module.' 前缀
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("  ✅ Weights loaded successfully")
    else:
        print(f"  ❌ Checkpoint NOT found: {args.checkpoint}")
        return

    model.eval()
    
    # 2. 封装导出模型
    export_model = ExportModel(model)
    export_model.eval()
    
    # 3. 创建 Dummy Input
    dummy_input = torch.randn(1, 3, args.height, args.width)
    
    # 4. 转换为 TorchScript
    print("\n🔄 Tracing model with TorchScript...")
    try:
        traced_model = torch.jit.trace(export_model, dummy_input)
        print("  ✅ Model traced successfully")
    except Exception as e:
        print(f"  ❌ Tracing failed: {e}")
        return
    
    # 5. 转换为 CoreML
    print("\n🍎 Converting to CoreML...")
    try:
        # 定义输入
        image_input = ct.ImageType(
            name="image",
            shape=(1, 3, args.height, args.width),
            scale=1/255.0/0.226,  # 简化的归一化 approximation or exact
            bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225], # TODO: 精确计算
        )
        # 精确的归一化参数 (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225])
        # CoreML 计算公式: output = (input * scale) + bias
        # PyTorch Normalize: output = (input - mean) / std = input * (1/std) - (mean/std)
        # 这一步比较复杂，通常建议在 CoreML convert 中指定 scale 和 bias
        
        scale = 1 / (0.226 * 255.0) # 简化近似，因为 std 接近 0.226
        # 更精确的做法是为每个通道指定 scale，但 ct.ImageType 通常仅支持单一 scale
        # 如果需要精确的每通道归一化，建议在模型内部实现预处理层
        
        # 为了兼容性，我们这里先使用简单的标准化配置
        # 并提示用户在 iOS 端做预处理，或者使用更高级的转换配置
        
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(
                    name="image", 
                    shape=(1, 3, args.height, args.width),
                )
            ],
            outputs=[ct.TensorType(name="feature")],
            minimum_deployment_target=ct.target.iOS16, # 支持较新的特性
        )
        
        # 添加元数据
        mlmodel.author = "Orion-Lite Team"
        mlmodel.license = "Academic"
        mlmodel.short_description = "MobileNetV4 based Person Re-ID"
        mlmodel.version = "2.0"
        
        # 保存
        mlmodel.save(args.output)
        print(f"  ✅ CoreML model saved to: {args.output}")
        
    except Exception as e:
        print(f"  ❌ CoreML conversion failed: {e}")
        return
    
    # 6. 验证 (可选)
    print("\n🔍 Validating conversion...")
    try:
        # PyTorch 输出
        with torch.no_grad():
            pt_out = export_model(dummy_input).numpy()
            
        # CoreML 输出
        pixel_input = dummy_input.numpy() # (1, 3, 256, 128)
        # CoreML 预测需要 dict
        prediction = mlmodel.predict({'image': pixel_input})
        ml_out = prediction['feature']
        
        # 计算误差
        diff = np.abs(pt_out - ml_out).max()
        print(f"  Max absolute difference: {diff:.6f}")
        
        if diff < 1e-3:
            print("  ✅ Consistency check passed!")
        else:
            print("  ⚠️ Consistency check warning: Difference might be too large.")
            
    except Exception as e:
        print(f"  ⚠️ Validation skipped/failed: {e}")

    print("\n" + "=" * 60)
    print("🎉 Export Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
