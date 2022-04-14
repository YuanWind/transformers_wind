from transformers import BertTokenizer
import tokenizers

# 制作自己的tokenizer
bwpt = tokenizers.BertWordPieceTokenizer()
filepath = "data_transformers_wind/train_data/unlabeled_train_data.txt" # 和本文第一部分的语料格式一致
bwpt.train(
    files=[filepath],
    vocab_size=800000,
    min_frequency=1,
    limit_alphabet=1000
)
bwpt.save_model('data_transformers_wind/pretrained_models1/') # 得到vocab.txt


token_path1='data_transformers_wind/my_nezha_cn/vocab.txt'
tokenizer1 =  BertTokenizer.from_pretrained(token_path1, do_lower_case=True)
token_path2='data_transformers_wind/pretrained_models1/vocab.txt'
tokenizer2 =  BertTokenizer.from_pretrained(token_path2, do_lower_case=True)
text = "笔记本电脑13141516"
tokens1 = tokenizer1.tokenize(text)
tokens2 = tokenizer2.tokenize(text)

print(tokens1)
print(tokens2)