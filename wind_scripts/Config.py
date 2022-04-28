# -*- encoding: utf-8 -*-
'''
@File    :   Config.py
@Time    :   2022/04/18 09:01:00
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
logger = logging.getLogger(__name__.replace('_', ''))
import json
from transformers import TrainingArguments
import os
from configparser import ConfigParser, ExtendedInterpolation
from dataclasses import dataclass, field

@dataclass
class MyTrainingArguments(TrainingArguments):
    
    adversarival_type: str = field(
        default= None,
        metadata={"help": "[None,'fgm','pgd']."},
    )
    fgm_e: float = field(
        default=1.0, metadata={"help": "FGM epsilon."}
    )
    pgd_e: float = field(
        default=1.0, metadata={"help": "PGD epsilon."}
    )
    pgd_a: float = field(
        default=1.0, metadata={"help": "PGD alpha."}
    )
    pgd_k: int = field(
        default=3, metadata={"help": "PGD's K."}
    )
    emb_name: str = field(
        default='emb', metadata={"help": "对那个embedding进行扰动."}
    )
    
    
class MyConfigs():
    def __init__(self, config_file, extra_args=None):
        """
        初始化config，优先级为 命令行 > config file > 默认值，可以随意在config文件或者命令行里加自定义的参数。
        Args:
            config_file (_type_): _description_
            extra_args (_type_, optional): _description_. Defaults to None.
        """
        self.config_file = config_file
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(config_file, encoding="utf-8")
        if extra_args:  # 如果命令行中有参数与config中的相同，则值使用命令行中传入的参数值
            extra_args = extra_args = dict([(k[2:], v) for k, v in zip(extra_args[::2], extra_args[1::2])])
            for section in config.sections():
                for k, v in config.items(section):
                    if k in extra_args:
                        v = type(v)(extra_args[k])
                        config.set(section, k, v)
            if len(extra_args)>0:
                if 'CMD' not in config.sections():
                    config.add_section('CMD')
                for k,v in extra_args.items():
                    config.set('CMD',k,v)
                    
        self._config = config
        self.train_args_dict={}
        for section in config.sections():
            for k, v in config.items(section):
                v = self.get_type(v)
                if section == 'Trainer':
                    self.train_args_dict[k] = v
                self.__setattr__(k, v)

        self.post_init()

    def get_type(self, v):
        """
        设置值的类型
        """
        v = v.replace(' ', '')
        if v.lower() == 'true':
            v = True
        elif v.lower() == 'false':
            v = False
        elif v.lower() == 'none':
            v = None
        elif v == '[]':
            v = []
        elif len(v)>2 and v[0] == '[' and v[-1] == ']':
            v = v.replace('[', '')
            v = v.replace(']', '')
            v = v.split(',')
        else:
            try:
                v = eval(v)
            except Exception:
                v = v
        return v
        
        

    def post_init(self):

        if self.temp_dir is not None:
            self.temp_dir = os.path.expanduser(self.temp_dir)
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)

        self.output_dir = self.output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # self.best_model_dir = 
        if not os.path.exists(self.best_model_file.rsplit('/', 1)[0]):
            os.makedirs(self.best_model_file.rsplit('/', 1)[0])
            
        self.log_file = self.log_file
        if not os.path.exists(self.log_file.rsplit('/', 1)[0]):
            os.makedirs(self.log_file.rsplit('/', 1)[0])

        self.trainer_args = MyTrainingArguments(**self.train_args_dict)

    def save(self):
        logger.info(f'Loaded config file from {self.config_file} sucessfully.')
        self._config.write(open(f'{self.output_dir}/' + self.config_file.split('/')[-1], 'w'))

        logger.info(f"Write this config to {f'{self.output_dir}/' + self.config_file.split('/')[-1]} sucessfully.")

        out_str = '\n'
        for section in self._config.sections():
            for k, v in self._config.items(section):
                out_str +='{} = {}\n'.format(k,v)
        logger.info(out_str)
    
    def to_json_string(self):
        out_json = {}
        for section in self._config.sections():
            for k, v in self._config.items(section):
                out_json[k] = v
        return json.dumps(out_json)

if __name__ == '__main__':
    config = MyConfigs('debug.cfg')
    logging.warning(config.log_file)
