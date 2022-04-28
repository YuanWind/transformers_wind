# Transformers-wind

基于开源库[Transformers](https://github.com/huggingface/transformers) 定制日常使用的代码，transformers版本：4.18.0 , 备份的代码在 transformers 目录里

## Requirements
```
torch==1.8.0
```

## TODO
1. 交叉验证的划分和模型集成的代码(logits直接平均、过linear、和投票策略)；
2. 增加拼音嵌入的代码；
3. 分层学习率的设定；
4. 增加DDP训练
5. 修改Tokenizer
6. 
## 主要特征
1. trainer 里加上 FGM、PGD；
2. 
