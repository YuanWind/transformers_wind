# -*- encoding: utf-8 -*-
'''
@File    :   DataVocab.py
@Time    :   2022/04/18 09:01:51
@Author  :   Yuan Wind
@Desc    :   None
'''
from collections import defaultdict
import logging
logger = logging.getLogger(__name__.replace('_', ''))
from wind_scripts.utils import load_json, load_pkl, set_to_dict, set_to_orderedlist


class BaseVocab:
    def __init__(self):
        self.insts = []
        self.train_insts = []
        self.dev_insts = []
        self.test_insts = []
        
        self.label_set = set()
        self.label2id = {}
        self.id2label = []
        
    @property
    def num_labels(self):
        return len(self.id2label)
    
    def read_files(self, files, file_type = 'train'):
        """
        读取多个文件
        Args:
            files (list[str]): 要读取的文件列表
            file_type (str, optional): ['train','dev','test']. Defaults to 'train'.
        """
        for file in files:
            if file_type == 'train':
                self.train_insts.extend(self.read_file(file))
            elif file_type == 'dev':
                self.dev_insts.extend(self.read_file(file))
            elif file_type == 'test':
                self.test_insts.extend(self.read_file(file))
            self.insts.extend(self.read_file(file))
            
    def read_file(self, file_path):
        """读取单个文件的数据
        Args:
            file (str): 文件路径
            file_type (str, optional): ['train','dev','test']. Defaults to 'train'.
        """
        raise NotImplementedError
    
    def build(self):
        """
        根据全部的insts构建vocab
        """
        raise NotImplementedError
    
class TokenVocab(BaseVocab):
    def __init__(self):
        super(TokenVocab, self).__init__()
        self.id2bio = ['O','B','I']
        self.bio2id = {'O':0,'B':1,'I':2}

        self.id2cls = ['UNK']
        self.cls2id = {'UNK':0}
        self.title2type = {}
        self.cls2cnt = defaultdict(int)
        
        self.entity2types = defaultdict(set)
        
    @property
    def bio_num(self): 
        return len(self.id2bio)
    
    @property
    def cls_num(self): 
        return len(self.id2cls)
    
    def read_file(self, file_path):
        if file_path[-3:] == 'pkl':
            return load_pkl(file_path)
        elif file_path[-3:] == 'son':
            return load_json(file_path)
    
    def build(self, insts):
        assert len(insts) != 0, '当前无数据，请先读取数据，再构建Vocab！'
        for data in insts:
            labels = data.get('labels')
            for lbl in labels:
                tp = lbl['type']
                entity = lbl['entity']
                self.label_set.add(tp)
                self.entity2types[entity].add(tp)
        self.id2label = set_to_orderedlist(self.label_set)
        self.label2id = set_to_dict(self.label_set)
        
    def build_other_vocab(self):
        pass