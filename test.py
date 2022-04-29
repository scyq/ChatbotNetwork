from transformers import BertTokenizer

gpt_model_name = "uer/gpt2-chinese-cluecorpussmall"
g_gpt_tokenizer = BertTokenizer.from_pretrained(gpt_model_name)

print(g_gpt_tokenizer)