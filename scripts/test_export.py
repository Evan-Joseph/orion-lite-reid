#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreML 导出测试脚本

创建一个临时检查点，验证 export_coreml.py 是否能正常工作。
"""

import sys
import os
import torch

# 添加项目根目录到路径
project_root = '/Users/joseph/Desktop/【人工智能技术结课项目】ReID重识别'
sys.path.insert(0, project_root)

from orion_lite.modeling import OrionLite


def test_export():
    print("=" * 60)
    print("Testing CoreML Export Pipeline")
    print("=" * 60)
    
    # 1. 创建临时模型和检查点
    print("\n📦 Creating dummy checkpoint...")
    model = OrionLite(backbone_name='mobilenetv4_conv_small', num_classes=10)
    
    checkpoint_path = 'temp_checkpoint.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'best_rank1': 0.0,
    }, checkpoint_path)
    print(f"  Saved to {checkpoint_path}")
    
    # 2. 运行导出脚本
    print("\n🚀 Running export script...")
    output_model = 'OrionLite_Test.mlpackage'
    
    cmd = (
        f"python {project_root}/tools/deploy/export_coreml.py "
        f"--checkpoint {checkpoint_path} "
        f"--output {output_model} "
        f"--num_classes 10"  # 注意这里要匹配
    )
    
    ret = os.system(cmd)
    
    # 3. 验证结果
    success = False
    if ret == 0 and os.path.exists(output_model):
        print(f"\n✅ Export successful: {output_model}")
        success = True
    else:
        print("\n❌ Export failed!")
    
    # 4. 清理
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists(output_model):
        import shutil
        shutil.rmtree(output_model)
        
    print("\n" + "=" * 60)
    return success


if __name__ == "__main__":
    test_export()
