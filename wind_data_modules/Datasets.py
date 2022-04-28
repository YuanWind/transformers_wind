# -*- encoding: utf-8 -*-
'''
@File    :   Datasets.py
@Time    :   2022/04/18 09:01:31
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
logger = logging.getLogger(__name__.replace('_', ''))
from torch.utils.data import Dataset
import torch
import jieba

class BaseDataset(Dataset):
    def __init__(self, config, insts, data_type = 'train'):
        """
        data_type: all, trian, dev, test
        """
        super().__init__()
        self.config = config
        self.data_type = data_type
        self.insts = insts
        self.items = None
        
    def __len__(self):
        """
        如果子类不实现该方法，那么Trainer里边默认的Dataloader将不会进行sample，否则会进行randomsample
        TODO 不实现的话有点问题
        """
        return len(self.items)
    
    def __getitem__(self,index) :
        """
        请与Datacollator进行联合设计出模型的batch输入，建议利用字典进行传参。
        """
        return self.items[index]
class NERDataset(BaseDataset):
    def __init__(self, config, insts, tokenizer, data_type='train', convert_here=False):
        super().__init__(config, insts, data_type)
        self.tokenizer = tokenizer
        self.items = convert_to_features(tokenizer, insts) if convert_here else self.insts
    
    
def convert_to_features(tokenizer, insts, config, vocab):
    features = []

    return features


    # def __len__(self):
    #     return len(self.insts)
    
    # def __getitem__(self,index):
    #     return self.items[index]
    