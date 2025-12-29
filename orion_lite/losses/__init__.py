# -*- coding: utf-8 -*-
"""损失函数模块"""

from .circle_loss import CircleLoss, CircleLossPairwise
from .triplet_loss import TripletLoss
from .cross_entropy import CrossEntropyLabelSmooth, SoftmaxTripletLoss

__all__ = [
    'CircleLoss',
    'CircleLossPairwise',
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftmaxTripletLoss',
]
