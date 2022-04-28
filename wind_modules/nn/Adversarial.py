# -*- encoding: utf-8 -*-
'''
@File    :   Adversarial.py
@Time    :   2022/04/18 09:03:12
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging
logger = logging.getLogger(__name__.replace('_', ''))
import torch

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
class FGM():
    """
    基于FGM算法的攻击机制

    Args:
        module (:obj:`torch.nn.Module`): 模型

    Examples::

        >>> # 初始化
        >>> fgm = FGM(module)
        >>> for batch_input, batch_label in data:
        >>>     # 正常训练
        >>>     loss = module(batch_input, batch_label)
        >>>     loss.backward() # 反向传播，得到正常的grad
        >>>     # 对抗训练
        >>>     fgm.attack() # 在embedding上添加对抗扰动
        >>>     loss_adv = module(batch_input, batch_label)
        >>>     loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        >>>     fgm.restore() # 恢复embedding参数
        >>>     # 梯度下降，更新参数
        >>>     optimizer.step()
        >>>     optimizer.zero_grad()

    Reference:
        [1]  https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, module):
        self.module = module
        self.backup = {}

    def attack(
        self,
        epsilon=1.,
        emb_name='emb'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(
        self,
        emb_name='emb'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    """
    基于PGD算法的攻击机制

    Args:
        module (:obj:`torch.nn.Module`): 模型

    Examples::

        >>> pgd = PGD(module)
        >>> K = 3
        >>> for batch_input, batch_label in data:
        >>>     # 正常训练
        >>>     loss = module(batch_input, batch_label)
        >>>     loss.backward() # 反向传播，得到正常的grad
        >>>     pgd.backup_grad()
        >>>     # 对抗训练
        >>>     for t in range(K):
        >>>         pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
        >>>         if t != K-1:
        >>>             optimizer.zero_grad()
        >>>         else:
        >>>             pgd.restore_grad()
        >>>         loss_adv = module(batch_input, batch_label)
        >>>         loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        >>>     pgd.restore() # 恢复embedding参数
        >>>     # 梯度下降，更新参数
        >>>     optimizer.step()
        >>>     optimizer.zero_grad()

    Reference:
        [1]  https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, module):
        self.module = module
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(
        self,
        epsilon=1.,
        alpha=0.3,
        emb_name='emb',
        is_first_attack=False
    ):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                if  param.grad is not None:
                    self.grad_backup[name] = param.grad.clone()
                # else:
                #     logger.warning(f'{name} 的 param 没有梯度！')

    def restore_grad(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad and name in self.grad_backup:
                param.grad = self.grad_backup[name]