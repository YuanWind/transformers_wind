# -*- encoding: utf-8 -*-
'''
@File    :   Trainer.py
@Time    :   2022/04/18 09:03:32
@Author  :   Yuan Wind
@Desc    :   None
'''
import logging

from wind_scripts.Evaluater import Evaluater
logger = logging.getLogger(__name__.replace('_', ''))
from transformers.trainer import *
from wind_modules.nn.Adversarial import FGM, PGD

class MyTrainer(Trainer):
    def __init__(self, model, config, **kwargs):
        super().__init__(model, args=config.trainer_args, **kwargs)
        self.config = config
        self.adversarival = None
        
        if self.args.adversarival_type == 'fgm': # 在这里初始化，如果是分布式训练，会不会有影响？
            self.adversarival = FGM(model) 
            logger.info(f'------------Use FGM, fgm_e = {self.args.fgm_e}, emb_name = {self.args.emb_name}.---------------')
        elif self.args.adversarival_type == 'pgd':
            self.adversarival = PGD(model)
            logger.info(f'------------Use PGD, pgd_e: {self.args.pgd_e}, pdg_a:{self.args.pgd_a} , emb_name = {self.args.emb_name}.---------------')

    def loss_backward(self,loss):
        """
        非原Trainer有的
        """
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
            
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """重写了该方法，主要加入了 FGM、PGD
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
                self.adversarival.attack(epsilon=self.args.fgm_e, emb_name=self.args.emb_name)
                with self.autocast_smart_context_manager():
                    loss_adv = self.compute_loss(model, inputs)
                self.loss_backward(loss_adv)
                self.adversarival.restore(emb_name=self.args.emb_name)
            elif self.args.adversarival_type == 'pgd':
                self.adversarival.backup_grad()
                for t in range(self.args.pgd_k):
                    self.adversarival.attack(epsilon=self.args.pgd_e,
                                             alpha=self.args.pgd_a,
                                             emb_name=self.args.emb_name,
                                             is_first_attack=(t==0)
                                             )
                    if t != self.args.pgd_k-1:
                        self.optimizer.zero_grad()
                    else:
                        self.adversarival.restore_grad()
                    with self.autocast_smart_context_manager():
                        loss_adv = self.compute_loss(model, inputs)
                    self.loss_backward(loss_adv)
                self.adversarival.restore(emb_name=self.args.emb_name)
            loss = loss_adv
        
        return loss.detach()

    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ):
        """
        Trainer自带的该方法是将验证集或者测试集全部预测完之后，将全部的logits、labels收集起来。
        之后再传到Evaluater里计算metric。可能会浪费大量的显存或者内存。
        此处MyTrainer增加了每一个step的后处理操作。将一定的steps预测的结果写入文件而不是全部的,然后在evaluate里读取文件计算metric
        """
        args = self.args
        
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        
        batch_size = self.args.per_device_eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")
        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = inputs["input_ids"] if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                Evaluater.steps_evaluate(logits, inputs_decode, labels)
                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
        if labels_host is not None:
            labels = nested_numpify(labels_host)

        Evaluater.steps_evaluate(preds_host=logits, inputs_host=inputs_decode, labels_host=labels)        
        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        # if all_preds is not None:
        #     all_preds = nested_truncate(all_preds, num_samples)
        # if all_labels is not None:
        #     all_labels = nested_truncate(all_labels, num_samples)
        # if all_inputs is not None:
        #     all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None:
            metrics = self.compute_metrics()
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    # def get_train_dataloader(self):
    #     pass
    # get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        # pass
        
    # def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # pass