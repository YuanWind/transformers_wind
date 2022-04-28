# -*- encoding: utf-8 -*-
'''
@File    :   Loss_func.py
@Time    :   2022/04/27 17:38:37
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
import torch
logger = logging.getLogger(__name__.replace('_', ''))

def GPCE(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1) # 非标签类的得分
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1) # 标签类的得分
    
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return neg_loss + pos_loss