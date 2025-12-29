# -*- coding: utf-8 -*-
"""
Orion-Lite 默认配置

使用 YACS 进行配置管理，参考 FastReID 的配置风格。
"""

from yacs.config import CfgNode as CN


def get_cfg_defaults():
    """获取默认配置"""
    _C = CN()
    
    # ==================== 模型配置 ====================
    _C.MODEL = CN()
    _C.MODEL.DEVICE = "mps"  # mps / cuda / cpu
    _C.MODEL.NAME = "orion_lite"
    _C.MODEL.PRETRAINED = True
    _C.MODEL.PRETRAINED_PATH = ""
    
    # Backbone
    _C.MODEL.BACKBONE = CN()
    _C.MODEL.BACKBONE.NAME = "ghostnetv2_100"  # timm 模型名
    _C.MODEL.BACKBONE.PRETRAINED = True
    _C.MODEL.BACKBONE.LAST_STRIDE = 1  # 最后一层步长，1 保留更多空间信息
    
    # Head
    _C.MODEL.HEAD = CN()
    _C.MODEL.HEAD.NAME = "BNNeckHead"
    _C.MODEL.HEAD.FEAT_DIM = 512  # 输出特征维度
    _C.MODEL.HEAD.POOL_TYPE = "avg"  # avg / max / gem
    _C.MODEL.HEAD.NECK = "bnneck"  # bnneck / no
    _C.MODEL.HEAD.NECK_FEAT = "after"  # before / after (用于推理)
    
    # Attention
    _C.MODEL.ATTENTION = CN()
    _C.MODEL.ATTENTION.ENABLED = True
    _C.MODEL.ATTENTION.TYPE = "coord"  # coord / se / cbam
    _C.MODEL.ATTENTION.REDUCTION = 32
    
    # ==================== 数据配置 ====================
    _C.DATA = CN()
    _C.DATA.DATASET = "Market1501"
    _C.DATA.ROOT = "./data"
    
    # 输入尺寸
    _C.DATA.HEIGHT = 256
    _C.DATA.WIDTH = 128
    
    # 数据增强
    _C.DATA.AUGMENTATION = CN()
    _C.DATA.AUGMENTATION.RE_PROB = 0.5  # Random Erasing 概率
    _C.DATA.AUGMENTATION.PADDING = 10
    _C.DATA.AUGMENTATION.FLIP_PROB = 0.5
    _C.DATA.AUGMENTATION.JITTER = 0.0  # Color Jitter
    
    # 归一化参数 (ImageNet)
    _C.DATA.NORM = CN()
    _C.DATA.NORM.MEAN = [0.485, 0.456, 0.406]
    _C.DATA.NORM.STD = [0.229, 0.224, 0.225]
    
    # ==================== 训练配置 ====================
    _C.TRAIN = CN()
    _C.TRAIN.EPOCHS = 60
    _C.TRAIN.BATCH_SIZE = 64  # M1 Pro 16GB 建议 64-128
    _C.TRAIN.NUM_WORKERS = 4
    
    # 采样策略
    _C.TRAIN.SAMPLER = CN()
    _C.TRAIN.SAMPLER.NAME = "pk"  # pk / random
    _C.TRAIN.SAMPLER.NUM_INSTANCES = 4  # PK采样：每个ID采样K张
    
    # ==================== 损失函数配置 ====================
    _C.LOSS = CN()
    
    # ID Loss (Cross Entropy)
    _C.LOSS.ID = CN()
    _C.LOSS.ID.ENABLED = True
    _C.LOSS.ID.LABEL_SMOOTH = 0.1
    _C.LOSS.ID.SCALE = 1.0
    
    # Triplet Loss
    _C.LOSS.TRIPLET = CN()
    _C.LOSS.TRIPLET.ENABLED = True
    _C.LOSS.TRIPLET.MARGIN = 0.3
    _C.LOSS.TRIPLET.HARD_MINING = True
    _C.LOSS.TRIPLET.SCALE = 1.0
    
    # ==================== 优化器配置 ====================
    _C.SOLVER = CN()
    _C.SOLVER.OPTIMIZER = "adam"  # adam / sgd / adamw
    _C.SOLVER.BASE_LR = 3.5e-4
    _C.SOLVER.WEIGHT_DECAY = 5e-4
    _C.SOLVER.MOMENTUM = 0.9  # for SGD
    
    # 学习率调度
    _C.SOLVER.LR_SCHEDULER = CN()
    _C.SOLVER.LR_SCHEDULER.NAME = "cosine"  # step / cosine / warmup_cosine
    _C.SOLVER.LR_SCHEDULER.WARMUP_EPOCHS = 10
    _C.SOLVER.LR_SCHEDULER.WARMUP_FACTOR = 0.01
    _C.SOLVER.LR_SCHEDULER.STEPS = [30, 50]  # for step scheduler
    _C.SOLVER.LR_SCHEDULER.GAMMA = 0.1
    
    # 混合精度 (MPS 上建议关闭)
    _C.SOLVER.FP16 = False
    
    # ==================== 评估配置 ====================
    _C.TEST = CN()
    _C.TEST.BATCH_SIZE = 128
    _C.TEST.EVAL_PERIOD = 5  # 每隔多少 epoch 评估一次
    _C.TEST.RE_RANKING = False  # Re-ranking 后处理
    _C.TEST.FEAT_NORM = True  # 特征归一化
    
    # ==================== 输出配置 ====================
    _C.OUTPUT = CN()
    _C.OUTPUT.DIR = "./logs"
    _C.OUTPUT.CHECKPOINT_DIR = "./checkpoints"
    _C.OUTPUT.LOG_PERIOD = 50  # 每隔多少 iter 打印日志
    _C.OUTPUT.SAVE_PERIOD = 10  # 每隔多少 epoch 保存模型
    
    return _C


def merge_from_file(cfg, config_file):
    """从 YAML 文件合并配置"""
    cfg.merge_from_file(config_file)
    return cfg
