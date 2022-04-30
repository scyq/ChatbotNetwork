from transformers import BertTokenizer

bert_model_name = "bert-base-chinese"
gpt_model_name = "uer/gpt2-chinese-cluecorpussmall"

g_bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
g_gpt_tokenizer = BertTokenizer.from_pretrained(gpt_model_name)

print("g_bert_tokenizer", g_bert_tokenizer)
print("g_gpt_tokenizer", g_gpt_tokenizer)
print(g_bert_tokenizer.pad_token_id)
print(g_gpt_tokenizer.pad_token_id)