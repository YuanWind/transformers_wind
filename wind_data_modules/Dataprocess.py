# -*- encoding: utf-8 -*-
'''
@File    :   DataUtils.py
@Time    :   2022/04/18 09:01:43
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
logger = logging.getLogger(__name__.replace('_', ''))
import sys
sys.path.extend(['../../','../','./'])

from wind_scripts.utils import dump_json, dump_pkl, sort_dict
from wind_data_modules.DataVocab import TokenVocab

def build_vocab(train_files,dev_files,test_file,save_path):
    vocab = TokenVocab()
    vocab.read_files(train_files,file_type= 'train')
    vocab.read_files(dev_files,file_type= 'dev')
    vocab.read_files(test_file,file_type= 'test')
    vocab.build(vocab.train_insts+vocab.dev_insts) # 使用哪些数据构建 Vocab
    dump_pkl(vocab, save_path)
    
if __name__ == "__main__":
    pass