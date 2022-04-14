from transformers.trainer import *
from wind_modules.Adversarial import FGM, PGD


logger = logging.get_logger(__name__.replace('_', ''))

class MyTrainer(Trainer):
    def __init__(self, model, configs, **kwargs):
        super().__init__(model, args=configs.trainer_args, **kwargs)
        self.my_config = configs
        self.adversarival = None
        
        if self.args.adversarival_type == 'fgm': # 在这里初始化，如果是分布式训练，会不会有影响？
            self.adversarival = FGM(model) 
            logger.info(f'------------Use FGM, fgm_e = {self.args.fgm_e} .---------------')
        elif self.args.adversarival_type == 'pgd':
            self.adversarival = PGD(model)
            logger.info(f'------------Use PGD, pgd_e: {self.args.pgd_e}, pdg_a:{self.args.pgd_a} .---------------')
    
    
    def loss_backward(self,loss):
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
            
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs. 加入对抗训练过程。
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        self.loss_backward(loss)
        
        if self.adversarival is not None:
            if self.args.adversarival_type == 'fgm':
                self.adversarival.attack()
                with self.autocast_smart_context_manager():
                    loss_adv = self.compute_loss(model, inputs)
                self.loss_backward(loss_adv)
                self.adversarival.restore()
            elif self.args.adversarival_type == 'pgd':
                self.adversarival.backup_grad()
                for t in range(self.args.pgd_K):
                    self.adversarival.attack(is_first_attack=(t==0))
                    if t != self.args.pgd_K-1:
                        self.optimizer.zero_grad()
                    else:
                        self.adversarival.restore_grad()
                    with self.autocast_smart_context_manager():
                        loss_adv = self.compute_loss(model, inputs)
                    self.loss_backward(loss_adv)
                self.adversarival.restore()
            loss = loss_adv
        
        return loss.detach()


    # def get_train_dataloader(self):
    #     pass
    # get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        # pass
        
    # def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # pass