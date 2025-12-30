# -*- coding: utf-8 -*-
"""
Orion-Lite 行人重识别演示系统

功能：
1. 上传目标人物图片
2. 在视频或摄像头画面中实时检测并识别该人物
3. 支持多目标追踪和匹配

使用方法：
    python demo/app.py
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Optional
import time

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from ultralytics import YOLO

from orion_lite.modeling import OrionLite
from orion_lite.data.transforms import build_transforms


class ReIDDemo:
    """行人重识别演示系统"""
    
    def __init__(
        self,
        reid_checkpoint: str = "checkpoints/best.pth",
        yolo_model: str = "yolov8n.pt",
        device: str = "auto",
        similarity_threshold: float = 0.5,
    ):
        # 设备选择
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🖥️ Device: {self.device}")
        
        # 加载 YOLO 检测器
        print("📦 Loading YOLO detector...")
        self.detector = YOLO(yolo_model)
        
        # 加载 ReID 模型
        print("🏗️ Loading Orion-Lite ReID model...")
        self.reid_model = OrionLite(
            backbone_name='mobilenetv4_conv_small',
            pretrained=False,
            feat_dim=512,
            num_classes=751,
        )
        
        # 加载检查点
        if os.path.exists(reid_checkpoint):
            checkpoint = torch.load(reid_checkpoint, map_location='cpu', weights_only=False)
            self.reid_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from {reid_checkpoint}")
        else:
            print(f"⚠️ Checkpoint not found: {reid_checkpoint}")
        
        self.reid_model = self.reid_model.to(self.device)
        self.reid_model.eval()
        
        # 图像变换
        self.transform = build_transforms(height=256, width=128, is_train=False)
        
        # 目标特征库
        self.target_features: List[np.ndarray] = []
        self.target_images: List[np.ndarray] = []
        # 移除 threshold, 使用 Top-1 强制匹配
        
        print("✅ Demo system initialized (Top-1 Mode)!")
    
    def extract_feature(self, image: np.ndarray) -> np.ndarray:
        """提取单张图片的 ReID 特征"""
        # 转换为 PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 预处理
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            feat = self.reid_model(img_tensor)
            feat = F.normalize(feat, p=2, dim=1)
        
        return feat.cpu().numpy()[0]
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测画面中的行人"""
        results = self.detector(frame, classes=[0], verbose=False)  # class 0 = person
        
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                if conf > 0.5:  # 置信度阈值
                    boxes.append((x1, y1, x2, y2))
        
        return boxes
    
    def add_target(self, image: np.ndarray) -> str:
        """添加目标人物图片"""
        if image is None:
            return "❌ 请上传图片"
        
        # 提取特征
        feature = self.extract_feature(image)
        self.target_features.append(feature)
        self.target_images.append(image)
        
        return f"✅ 已添加目标人物 #{len(self.target_features)}"
    
    def clear_targets(self) -> str:
        """清空目标库"""
        self.target_features = []
        self.target_images = []
        return "🗑️ 已清空所有目标"
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧图像 (Top-1 匹配)"""
        if frame is None:
            return None
        
        boxes = self.detect_persons(frame)
        if len(boxes) == 0:
            return frame

        if len(self.target_features) == 0:
            # 没有目标，只画灰色框
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
            return frame
        
        # 1. 提取当前帧所有人的特征
        frame_feats = []
        for (x1, y1, x2, y2) in boxes:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                frame_feats.append(np.zeros(512)) # Padding
            else:
                frame_feats.append(self.extract_feature(crop))
                
        # 2. 构建相似度矩阵 (Rows=Detections, Cols=Targets)
        # 寻找全局最佳匹配
        assignments = {} # det_idx -> target_idx
        
        if len(frame_feats) > 0 and len(self.target_features) > 0:
            sim_matrix = np.zeros((len(frame_feats), len(self.target_features)))
            
            for i, f_feat in enumerate(frame_feats):
                for j, t_feat in enumerate(self.target_features):
                    sim_matrix[i, j] = float(np.dot(f_feat, t_feat))
            
            # 贪心匹配
            assigned_dets = set()
            assigned_targets = set()
            
            while True:
                if sim_matrix.size == 0: break
                # 找最大值
                if np.max(sim_matrix) == -1: break
                
                ind = np.unravel_index(np.argmax(sim_matrix, axis=None), sim_matrix.shape)
                max_sim = sim_matrix[ind]
                
                det_idx = ind[0]
                target_idx = ind[1]
                
                # 确保唯一性
                if det_idx not in assigned_dets and target_idx not in assigned_targets:
                    assignments[det_idx] = (target_idx, max_sim)
                    assigned_dets.add(det_idx)
                    assigned_targets.add(target_idx)
                
                # 屏蔽
                sim_matrix[det_idx, :] = -1
                sim_matrix[:, target_idx] = -1

        # 3. 绘制
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if i in assignments:
                # 匹配到目标
                tid, sim = assignments[i]
                color = (0, 255, 0)
                thick = 4
                label = f"Target #{tid+1}: {sim:.1%}"
                
                # 标签背景
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                cv2.putText(frame, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            else:
                # 路人
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
        
        return frame
    
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """处理上传的图片"""
        return self.process_frame(image)
    
    def process_video(self, video_path: str) -> str:
        """处理视频文件"""
        if video_path is None:
            return None
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 输出视频
        output_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            processed = self.process_frame(frame)
            out.write(processed)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        
        cap.release()
        out.release()
        
        return output_path


def create_demo():
    """创建 Gradio 演示界面"""
    
    # 初始化演示系统
    demo_system = ReIDDemo()
    
    # CSS 样式
    custom_css = """
    .gradio-container {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: white;
    }
    .stats-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Orion-Lite ReID Demo") as demo:
        # 标题
        gr.HTML("""
        <div class="header">
            <h1>🌟 Orion-Lite 行人重识别演示系统</h1>
            <p>上传目标人物图片,在视频或图片中识别该人物</p>
            <p style="font-size: 0.9em; opacity: 0.9;">
                Rank-1: 86.58% | mAP: 80.06% | 参数量: 3.46M
            </p>
        </div>
        """)
        
        with gr.Row():
            # 左侧：目标管理
            with gr.Column(scale=1):
                gr.Markdown("## 📷 目标人物")
                
                target_image = gr.Image(label="上传目标图片", type="numpy")
                
                with gr.Row():
                    add_btn = gr.Button("➕ 添加目标", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                
                status_text = gr.Textbox(label="状态", interactive=False)
                
                gr.Markdown("""
                ### 使用说明
                1. 上传目标人物的**完整身体照片**
                2. 可添加多个目标人物
                3. 上传待识别的图片或视频
                4. 系统会自动检测并匹配
                """)
            
            # 右侧：识别区域
            with gr.Column(scale=2):
                gr.Markdown("## 🔍 实时识别")
                
                with gr.Tabs():
                    # 图片识别
                    with gr.TabItem("📸 图片识别"):
                        input_image = gr.Image(label="上传待识别图片", type="numpy")
                        output_image = gr.Image(label="识别结果")
                        process_img_btn = gr.Button("🚀 开始识别", variant="primary")
                    
                    # 视频识别
                    with gr.TabItem("🎬 视频识别"):
                        input_video = gr.Video(label="上传视频")
                        output_video = gr.Video(label="识别结果")
                        process_vid_btn = gr.Button("🚀 处理视频", variant="primary")
                    
                    # 摄像头实时
                    with gr.TabItem("📹 实时摄像头"):
                        gr.Markdown("""
                        ### 摄像头实时识别
                        
                        由于 Gradio 的限制，实时摄像头功能需要单独运行：
                        
                        ```bash
                        python demo/webcam_demo.py
                        ```
                        
                        按 **Q** 键退出
                        """)
        
        # 事件绑定
        add_btn.click(
            fn=demo_system.add_target,
            inputs=[target_image],
            outputs=[status_text]
        )
        
        clear_btn.click(
            fn=demo_system.clear_targets,
            inputs=[],
            outputs=[status_text]
        )
        
        process_img_btn.click(
            fn=demo_system.process_image,
            inputs=[input_image],
            outputs=[output_image]
        )
        
        process_vid_btn.click(
            fn=demo_system.process_video,
            inputs=[input_video],
            outputs=[output_video]
        )
        
        # 底部信息
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; 
                    background: #f8f9fa; border-radius: 10px;">
            <p><strong>Orion-Lite v2.0</strong> | 
               MobileNetV4 + Coordinate Attention + Circle Loss</p>
            <p style="font-size: 0.9em; color: #666;">
               人工智能技术结课项目 | Market-1501 Dataset
            </p>
        </div>
        """)
    
    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
