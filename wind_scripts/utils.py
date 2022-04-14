import contextlib
import logging
import os
import random
import pickle
import json


def sort_dict(d, mode='k', reverse=False): 
    """对字典按照key或者value排序

    Args:
        d (dict): 待排序的字典对象
        mode (str, optional): 'k'-->键排序, 'v'-->值排序 . Defaults to 'k'.
        reverse (bool, optional): True为降序排列. Defaults to False.

    Returns:
        list(touple): 返回一个list, 里边touple第一个为key, 第二个为value
    """
    # assert type(d) == dict, 'sort_dict仅支持对dict排序, 当前对象为:{}'.format(type(d))
    if mode == 'k': 
        return [(i, d[i]) for i in sorted(d, reverse=reverse)]
    elif mode == 'v': 
        return sorted(d.items(), key = lambda kv: kv[1], reverse=reverse)
    else:
        print('排序失败')
        return d

def check_empty_gpu(find_ours=11.5):
    
    import pynvml
    import time
    start = time.time()
    find_times = 0
    pynvml.nvmlInit()
    cnt = pynvml.nvmlDeviceGetCount()
    while True:
        for i in range(cnt):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if info.used < 100*1000000:  # 100 M         
                return i
        cur_time = time.time()
        during = int(cur_time-start)+1
        if  during % 1800 == 0:
            find_times+=0.5
            print(f'已经经过{find_times}小时，还未找到GPU')
        
        if find_times > find_ours: # 如果超过 find_ours 个小时还没有分配到GPU，则停止程序
            print(f'已经经过{find_times}小时，还未找到GPU，终止程序')
            exit()     
        
def set_logger(to_console=True, log_file=''):
    logger = logging.getLogger()  # 不加名称设置root logger
    level = logging.INFO
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # 使用FileHandler输出到文件
    if log_file != '':
        fh = logging.FileHandler(log_file)
        set_leval(fh, level, formatter, logger)
    # 使用StreamHandler输出到屏幕
    if to_console:
        ch = logging.StreamHandler()
        set_leval(ch, level, formatter, logger)

def set_leval(log_hander, level, formatter, logger):
    log_hander.setLevel(level)
    log_hander.setFormatter(formatter)
    logger.addHandler(log_hander)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    with contextlib.suppress(Exception):
        import numpy as np
        np.random.seed(seed)
        
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def dump_pkl(data, f_name):
    with open(f_name, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)

def dump_json(data, f_name):
    with open(f_name, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def load_json(f_name):
    with open(f_name, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    return data