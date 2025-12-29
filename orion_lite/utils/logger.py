# -*- coding: utf-8 -*-
"""
日志工具 (Logger)

提供统一的 logging 接口，支持同时输出到控制台和文件。
"""

import sys
import os
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def setup_logger(output_dir: str = None, name: str = "OrionLite"):
    """
    初始化 Logger
    
    Args:
        output_dir: 日志保存目录
        name: Logger 名称
        
    Returns:
        logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # 格式化
    fmt = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
    date_fmt = "%m/%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)
    
    # 控制台处理器
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # 文件处理器
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "train_log.txt")
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
