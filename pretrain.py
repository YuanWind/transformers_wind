from transformers import  BertTokenizer, WEIGHTS_NAME,TrainingArguments
from wind_modules.models.modeling_nezha import NeZhaForSequenceClassification,NeZhaForMaskedLM
from wind_modules.models.configuration_nezha import NeZhaConfig
import tokenizers
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
    LineByLineTextDataset
)
from wind_scripts.utils import dump_pkl, load_pkl

set_seed(666)
## 制作自己的tokenizer
# bwpt = tokenizers.BertWordPieceTokenizer()
# filepath = "data_transformers_wind/train_data/unlabeled_train_data.txt" # 和本文第一部分的语料格式一致
# bwpt.train(
#     files=[filepath],
#     vocab_size=20000,
#     min_frequency=1,
#     limit_alphabet=1000
# )
# bwpt.save_model('data_transformers_wind/pretrained_models/') # 得到vocab.txt

## 加载tokenizer和模型
model_path='data_transformers_wind/my_nezha_cn'
token_path='data_transformers_wind/pretrained_models/vocab.txt'
tokenizer =  BertTokenizer.from_pretrained(token_path, do_lower_case=True)
config=NeZhaConfig.from_pretrained(model_path)
model=NeZhaForMaskedLM.from_pretrained(model_path, config=config)
model.resize_token_embeddings(len(tokenizer))

print('Start to build dataset...')
# 通过LineByLineTextDataset接口 加载数据 #长度设置为128, # 这里file_path于本文第一部分的语料格式一致
# train_dataset=LineByLineTextDataset(tokenizer=tokenizer,file_path='data_transformers_wind/train_data/unlabeled_train_data.txt',block_size=128) 

# dump_pkl(train_dataset,'data_transformers_wind/outputs/train_dataset.pkl')

train_dataset=load_pkl('data_transformers_wind/outputs/train_dataset.pkl')
print('Start to build MLM模型的数据DataCollator...')
# MLM模型的数据DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
# 训练参数

pretrain_batch_size=128
num_train_epochs=10
training_args = TrainingArguments(
    output_dir='data_transformers_wind/outputs', 
    overwrite_output_dir=True, 
    num_train_epochs=num_train_epochs, 
    learning_rate=4e-5,
    save_strategy = 'epoch',
    per_device_train_batch_size=pretrain_batch_size,
    report_to = ['tensorboard'],
    save_total_limit=5) # save_steps=10000
# 通过Trainer接口训练模型
trainer = Trainer(
    model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset)

print('Start to train...')
# 开始训练
trainer.train()
trainer.save_model('data_transformers_wind/outputs/')

print('Finish train...')