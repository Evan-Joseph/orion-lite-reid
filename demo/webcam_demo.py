# -*- coding: utf-8 -*-
"""
Orion-Lite Demo Ultra - 终极演示版 (v4.0 - Top1 模式)

改进点:
1. Top-1 强匹配模式：仅高亮全局最相似的目标，无视阈值，拒绝误检
2. 移除阈值调节 UI
3. 增强的 IoU + ReID 混合追踪
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from orion_lite.modeling import OrionLite
from orion_lite.data.transforms import build_transforms


class SimpleTracker:
    def __init__(self, max_age=30):
        self.tracks = {} 
        self.next_id = 0
        self.max_age = max_age

    def update(self, detections, features):
        """
        detections: list of [x1, y1, x2, y2]
        """
        matched_indices = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        
        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            track_ids = list(self.tracks.keys())
            
            for i, tid in enumerate(track_ids):
                t_box = self.tracks[tid]['box']
                for j, d_box in enumerate(detections):
                    iou_matrix[i, j] = self._compute_iou(t_box, d_box)
            
            while True:
                if iou_matrix.size == 0: break
                ind = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
                max_iou = iou_matrix[ind]
                if max_iou < 0.3: break 
                
                tid = track_ids[ind[0]]
                det_idx = ind[1]
                
                matched_indices.append((tid, det_idx))
                if det_idx in unmatched_dets: unmatched_dets.remove(det_idx)
                if tid in unmatched_tracks: unmatched_tracks.remove(tid)
                
                iou_matrix[ind[0], :] = -1
                iou_matrix[:, ind[1]] = -1

        # 更新已匹配
        for tid, det_idx in matched_indices:
            self.tracks[tid]['box'] = detections[det_idx]
            self.tracks[tid]['feat'] = features[det_idx] # 更新特征
            self.tracks[tid]['age'] = 0
            cx, cy = (detections[det_idx][0]+detections[det_idx][2])//2, (detections[det_idx][1]+detections[det_idx][3])//2
            self.tracks[tid]['history'].append((cx, cy))
            
        # 创建新轨迹
        for det_idx in unmatched_dets:
            box = detections[det_idx]
            self.tracks[self.next_id] = {
                'box': box,
                'feat': features[det_idx],
                'age': 0,
                'history': deque(maxlen=40),
            }
            cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
            self.tracks[self.next_id]['history'].append((cx, cy))
            self.next_id += 1

        # 删除过期
        del_ids = []
        for tid in unmatched_tracks:
            self.tracks[tid]['age'] += 1
            if self.tracks[tid]['age'] > self.max_age:
                del_ids.append(tid)
        for tid in del_ids: del self.tracks[tid]
            
        return self.tracks

    def _compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

# ==========================================
# UI 组件
# ==========================================

class Button:
    def __init__(self, x, y, w, h, text, key_hint, callback):
        self.rect = (x, y, w, h)
        self.text = text
        self.key_hint = key_hint
        self.callback = callback
        self.is_hovered = False
        self.is_pressed = False
        self.click_feedback_timer = 0 

    def check_hover(self, mx, my):
        x, y, w, h = self.rect
        self.is_hovered = (x <= mx <= x + w) and (y <= my <= y + h)
        return self.is_hovered

    def trigger(self):
        self.is_pressed = True
        self.click_feedback_timer = 5 
        if self.callback:
            self.callback()

    def update(self):
        if self.click_feedback_timer > 0:
            self.click_feedback_timer -= 1
            if self.click_feedback_timer == 0:
                self.is_pressed = False

    def draw(self, draw_obj, font, small_font):
        x, y, w, h = self.rect
        
        if self.is_pressed or self.click_feedback_timer > 0:
            bg_color = (255, 255, 255, 200) 
            text_color = (0, 0, 0, 255)
            outline_color = (255, 255, 255, 255)
        elif self.is_hovered:
            bg_color = (255, 255, 255, 60)
            text_color = (255, 255, 255, 255)
            outline_color = (200, 200, 200, 150)
        else:
            bg_color = (0, 0, 0, 140)
            text_color = (220, 220, 220, 255)
            outline_color = (100, 100, 100, 100)
            
        draw_obj.rounded_rectangle([x, y, x+w, y+h], radius=8, fill=bg_color, outline=outline_color, width=1)
        
        bbox = font.getbbox(self.text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        k_bbox = small_font.getbbox(self.key_hint)
        kw, kh = k_bbox[2] - k_bbox[0], k_bbox[3] - k_bbox[1]
        
        total_h = th + kh + 4
        start_y = y + (h - total_h) // 2
        
        draw_obj.text((x + (w-tw)//2, start_y-2), self.text, font=font, fill=text_color)
        draw_obj.text((x + (w-kw)//2, start_y + th + 2), self.key_hint, font=small_font, fill=text_color)


class UIEngine:
    def __init__(self, width, height, font_path=None):
        self.width = width
        self.height = height
        self.buttons = []
        self.font = self._load_font(font_path, 18) 
        self.small_font = self._load_font(font_path, 12)
        
        self.fps_queue = deque(maxlen=150)
        self.mouse_x = 0
        self.mouse_y = 0
        
    def _load_font(self, path, size):
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "arial.ttf"
        ]
        for p in font_paths:
            if os.path.exists(p):
                try: return ImageFont.truetype(p, size, index=0)
                except: continue
        return ImageFont.load_default()

    def add_button(self, btn):
        self.buttons.append(btn)
        
    def update(self, mx, my):
        self.mouse_x, self.mouse_y = mx, my
        for btn in self.buttons:
            btn.check_hover(mx, my)
            btn.update() 

    def handle_click(self, x, y):
        for btn in self.buttons:
            if btn.check_hover(x, y):
                btn.trigger()
                return True
        return False

    def draw_graph(self, draw_obj, data, x, y, w, h, color, label):
        draw_obj.rounded_rectangle([x, y, x+w, y+h], radius=5, fill=(0, 0, 0, 100))
        if len(data) < 2: return
        max_v = max(max(data), 1)
        pts = []
        for i, v in enumerate(data):
            px = x + int(i * (w / len(data)))
            py = y + h - int((v / max_v) * (h - 10)) - 5
            pts.append((px, py))
        draw_obj.line(pts, fill=color, width=2)
        draw_obj.text((x+5, y+2), f"{label}: {data[-1]:.1f}", font=self.small_font, fill=color)


# ==========================================
# 主程序
# ==========================================

class DemoUltraV4:
    def __init__(self, reid_ckpt="checkpoints/best.pth", yolo="yolov8n.pt"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🚀 Device: {self.device}")
        
        self.detector = YOLO(yolo)
        self.reid = OrionLite(backbone_name='mobilenetv4_conv_small', feat_dim=512, num_classes=751)
        
        if os.path.exists(reid_ckpt):
            state = torch.load(reid_ckpt, map_location='cpu', weights_only=False)
            self.reid.load_state_dict(state['model_state_dict'])
            
        self.reid.to(self.device).eval()
        self.transform = build_transforms(height=256, width=128, is_train=False)
        
        self.targets = [] 
        self.tracker = SimpleTracker(max_age=15)
        
        self.mirror = True
        self.paused = False
        self.selecting = False
        self.quit = False
        
        self.ui = None

    def extract(self, cv_img):
        pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        t = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = F.normalize(self.reid(t), p=2, dim=1)
        return f.cpu().numpy()[0]

    def trigger_button_by_name(self, name):
        for btn in self.ui.buttons:
            if btn.text == name:
                btn.trigger()

    # Callbacks
    def cb_add(self):
        self.paused = True
        self.selecting = True
        print(">> 选择模式: 请点击目标")
        
    def cb_clear(self):
        self.targets = []
        self.tracker.tracks = {}
        
    def cb_mirror(self): self.mirror = not self.mirror
    def cb_quit(self): self.quit = True

    def init_ui(self, w, h):
        self.ui = UIEngine(w, h)
        bw, bh = 120, 55 
        bx = w - bw - 20
        sy = 60
        gp = 15
        
        self.ui.add_button(Button(bx, sy, bw, bh, "添加目标", "(A)", self.cb_add))
        self.ui.add_button(Button(bx, sy+(bh+gp)*1, bw, bh, "清空库", "(C)", self.cb_clear))
        self.ui.add_button(Button(bx, sy+(bh+gp)*2, bw, bh, "镜像开关", "(M)", self.cb_mirror))
        self.ui.add_button(Button(bx, h-80, bw, bh, "退出", "(Q)", self.cb_quit))

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        
        ret, frame = cap.read()
        if not ret: return
        h, w = frame.shape[:2]
        self.init_ui(w, h)
        
        win_name = "Orion-Lite Demo V4 (Top-1)"
        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
        
        def mouse(e, x, y, f, p):
            self.ui.update(x, y) 
            if e == cv2.EVENT_LBUTTONDOWN:
                if self.ui.handle_click(x, y): return
                if self.selecting: self._do_select(x, y)
                
        cv2.setMouseCallback(win_name, mouse)
        
        self.curr_boxes = []
        self.paused_frame = None
        
        while not self.quit:
            # 键盘快捷键
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'): self.trigger_button_by_name("退出")
            elif key == ord('a'): self.trigger_button_by_name("添加目标")
            elif key == ord('c'): self.trigger_button_by_name("清空库")
            elif key == ord('m'): self.trigger_button_by_name("镜像开关")
            
            if not self.paused:
                ret, frame = cap.read()
                if not ret: break
                if self.mirror: frame = cv2.flip(frame, 1)
                self.paused_frame = frame.copy()
                
                self.ui.update(self.ui.mouse_x, self.ui.mouse_y)
                
                # 1. 检测
                res = self.detector(frame, verbose=False, classes=[0], conf=0.5)
                
                dets = []
                feats = []
                self.curr_boxes = []
                
                if len(res) > 0:
                    for box in res[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0: continue
                        
                        dets.append([x1, y1, x2, y2])
                        feats.append(self.extract(crop))
                        self.curr_boxes.append((x1, y1, x2, y2))
                        
                # 2. 追踪更新
                tracks = self.tracker.update(dets, feats)
                
                # 3. Top-1 匹配
                # 策略：对于每个目标（Target），找到当前帧中最相似的 Track
                track_assignments = {} # track_id -> {'label': str, 'sim': float, 'color': tuple}
                
                # 如果有目标库
                if len(self.targets) > 0:
                    # 构建相似度矩阵：Rows=Tracks, Cols=Targets
                    track_ids = list(tracks.keys())
                    if len(track_ids) > 0:
                        sim_matrix = np.zeros((len(track_ids), len(self.targets)))
                        for i, tid in enumerate(track_ids):
                            feat = tracks[tid]['feat']
                            for j, (t_feat, _) in enumerate(self.targets):
                                sim_matrix[i, j] = float(np.dot(feat, t_feat))
                        
                        # 贪心匹配：每次取全局最大相似度
                        # 确保每个 Target 最多只出现在一个 Track 上
                        # 确保每个 Track 最多只被分配一个 Target
                        assigned_tracks = set()
                        assigned_targets = set()
                        
                        while True:
                            if sim_matrix.size == 0: break
                            if np.max(sim_matrix) == -1: break # 全匹配完
                            
                            ind = np.unravel_index(np.argmax(sim_matrix, axis=None), sim_matrix.shape)
                            max_sim = sim_matrix[ind]
                            
                            tid = track_ids[ind[0]]
                            target_idx = ind[1]
                            
                            if tid not in assigned_tracks and target_idx not in assigned_targets:
                                assigned_tracks.add(tid)
                                assigned_targets.add(target_idx)
                                
                                track_assignments[tid] = {
                                    'label': f"目标 #{target_idx+1}",
                                    'sim': max_sim,
                                    'color': (0, 255, 0) # Green
                                }
                            
                            # 屏蔽
                            sim_matrix[ind[0], :] = -1
                            sim_matrix[:, ind[1]] = -1

                # 4. 绘图
                lat = res[0].speed['inference']
                self.ui.fps_queue.append(1000/(lat+1e-5))
                
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil, 'RGBA')
                
                for tid, trk in tracks.items():
                    # 判断是否有身份
                    has_id = tid in track_assignments
                    
                    if has_id:
                        info = track_assignments[tid]
                        color = info['color']
                        label = info['label']
                        sim = info['sim']
                        
                        # 绘制轨迹
                        pts = list(trk['history'])
                        if len(pts) > 2:
                            draw.line(pts, fill=(0, 255, 0, 180), width=6)
                        
                        # 绘制强力框
                        box = trk['box']
                        draw.rectangle(box, outline=color, width=5)
                        
                        # 绘制标签
                        text = f"{label} {sim:.1%}"
                        tx, ty = box[0], box[1] - 32
                        draw.rounded_rectangle([tx, ty, tx+160, ty+30], radius=5, fill=(0,0,0,180))
                        draw.text((tx+10, ty+2), text, font=self.ui.font, fill=(255,255,255))
                    
                    else:
                        # 路人：极简模式
                        box = trk['box']
                        draw.rectangle(box, outline=(200, 200, 200), width=2)

                # UI
                self.ui.draw_graph(draw, list(self.ui.fps_queue), 20, h-80, 200, 50, (0, 255, 0), "FPS")
                for btn in self.ui.buttons: btn.draw(draw, self.ui.font, self.ui.small_font)
                
                info = f"目标数: {len(self.targets)}"
                draw.text((w-200, 40), info, font=self.ui.small_font, fill=(200,200,200))

                out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                cv2.imshow(win_name, out)
                
            else:
                if self.paused_frame is not None:
                    vis = self.paused_frame.copy()
                    overlay = vis.copy()
                    cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
                    vis = cv2.addWeighted(vis, 0.4, overlay, 0.6, 0)
                    for (x1, y1, x2, y2) in self.curr_boxes:
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(vis, "CLICK TO ADD", (w//2-150, h//2), 0, 1.2, (255,255,255), 2)
                    cv2.imshow(win_name, vis)
            
        cap.release()
        cv2.destroyAllWindows()

    def _do_select(self, x, y):
        for (bx1, by1, bx2, by2) in self.curr_boxes:
            if bx1 < x < bx2 and by1 < y < by2:
                crop = self.paused_frame[by1:by2, bx1:bx2]
                feat = self.extract(crop)
                self.targets.append((feat, crop))
                print(f"✅ 添加新目标 #{len(self.targets)}")
                break
        self.paused = False
        self.selecting = False

        
if __name__ == "__main__":
    app = DemoUltraV4()
    try:
        app.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
