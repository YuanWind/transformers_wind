# -*- encoding: utf-8 -*-
'''
@File    :   Evaluater.py
@Time    :   2022/04/18 09:00:08
@Author  :   Yuan Wind
@Desc    :   None
'''
import json
import logging
import numpy as np
from wind_scripts.utils import load_json, dump_json
logger = logging.getLogger(__name__.replace('_', ''))
class Evaluater:
    config = None
    stage = 'dev'
    vocab = None
    tokenizer = None
    finished_idx = 0
    res = []
    @staticmethod
    def evaluate():
        pass
    
    @staticmethod
    def steps_evaluate(preds_host, inputs_host, labels_host):
        pass
