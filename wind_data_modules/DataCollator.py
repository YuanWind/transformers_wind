# -*- encoding: utf-8 -*-
'''
@File    :   DataCollator.py
@Time    :   2022/04/18 09:01:19
@Author  :   Yuan Wind
@Desc    :   None
'''
from collections import defaultdict
import logging
from wind_data_modules.Datasets import  convert_to_features
logger = logging.getLogger(__name__.replace('_', ''))
import torch
from typing import Any

class BaseDatacollator:
    def __init__(self,config, vocab):
        self.config = config
        self.vocab = vocab
        
    def __call__(self, batch_data, convert_here=True):
        """子类实现该方法，作为Trainer的Dataloader的data_collator参数

        Args:
            batch_data (Any): 一个batch的 dataset 返回值
        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    

class NERDatacollator(BaseDatacollator):
    def __init__(self,config, tokenizer, vocab, convert_here=True):
        super().__init__(config, vocab)
        self.tokenizer = tokenizer
        self.convert_here = convert_here
    
    def __call__(self,batch_data):
        # sourcery skip: assign-if-exp, swap-if-expression
        if not self.convert_here:
            features = batch_data
        else:
            features = convert_to_features(self.tokenizer, batch_data, self.config, self.vocab)
            
        model_params = defaultdict(list)
        for one_item in features:
            for k,v in one_item.items():
                model_params[k].append(v)

        for k in model_params:
            if type(model_params[k][0]) is torch.Tensor:
                model_params[k] = torch.stack(model_params[k])
        return model_params

