# -*- encoding: utf-8 -*-
'''
@File    :   Main.py
@Time    :   2022/04/18 09:00:19
@Author  :   Yuan Wind
@Desc    :   使用流程如下
1. 完成 Dataprocess.py, 构建 Vocab；
2. 完成下面的 build_data 和 build_model 函数 即可
'''
import logging
logger = logging.getLogger(__name__.replace('_', ''))
import os
if 'OMP_NUM_THREADS' not in os.environ.keys():
    max_thread = 1
    os.environ['OMP_NUM_THREADS'] = str(max_thread) # 该代码最多可建立多少个进程， 要放在最前边才能生效。防止占用过多CPU资源
    logger.warning(f' 该程序最多可建立{max_thread}个线程。')


from argparse import ArgumentParser
import torch
from wind_scripts.Config import  MyConfigs
from wind_scripts.Trainer import MyTrainer
from wind_scripts.utils import set_logger, set_seed, check_empty_gpu
from wind_scripts.Evaluater import Evaluater

if 'CUDA_VISIBLE_DEVICES' not in os.environ.keys(): 
    gpu_number = check_empty_gpu()
    print(f' 未指定使用的GPU，将使用 {gpu_number} 卡。')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
    
def set_configs():
    argsParser = ArgumentParser()
    argsParser.add_argument('--config_file', type=str, default='wind_configs/debug.cfg')
    args, extra_args = argsParser.parse_known_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f'Config file {args.config_file} not found.')

    # 解析参数
    configs = MyConfigs(args.config_file, extra_args)

    # 设置 root logger
    set_logger(to_console=True, log_file=configs.log_file)
    logger.info(f"------------  Process ID {os.getpid()}, Process Parent ID {os.getppid()}  --------------------\n")
    configs.save()
    return configs

def main():   # sourcery skip: extract-duplicate-method
    config = set_configs()
    set_seed(config.seed)
    Evaluater.config = config
    vocab, train_set, dev_set, test_set, data_collator = build_data(config)
    model = build_model(config, vocab)
    trainer = MyTrainer(model, config, 
                        train_dataset=train_set, 
                        eval_dataset=dev_set, 
                        data_collator = data_collator,
                        compute_metrics=Evaluater.evaluate,
                        )

    if config.trainer_args.do_train and train_set is not None:
        logger.info('Start training...\n\n')
        trainer.train(config.resume_from_checkpoint)
        torch.save(model.state_dict(), config.best_model_file)
        logger.info(f'Finished training, save the last states to {config.best_model_file}\n\n')

    if config.trainer_args.do_eval and dev_set is not None:
        model.load_state_dict(torch.load(config.best_model_file))
        logger.info(f'Load model state from {config.best_model_file}')
        logger.info('Use loaded model to evaluate dev dataset:')
        Evaluater.stage = 'dev'
        trainer.evaluate(dev_set)
        
    if config.trainer_args.do_predict and test_set is not None:
        model.load_state_dict(torch.load(config.best_model_file))
        logger.info(f'Load model state from {config.best_model_file}')
        logger.info('Use loaded model to predict test dataset:')
        Evaluater.stage = 'test'
        trainer.predict(test_set)
        
    logger.info('---------------------------  Finish!  ----------------------------------\n\n')

def build_data(config):
    
    pass

def build_model(config,vocab):
    pass