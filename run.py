from train import *
import csv
from config import Config
from tqdm import tqdm

import os

opt = Config()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_device

TRAIN = opt.train_or_chat
delimiter = "\t"

print(f"We are on {'train' if TRAIN else 'chat'} mode.")


# print chatbot's words
def print_answer(sentences):
    # sentences = [sentence len, batch size, vocab dim]
    print("chatbot: ")

    sentence = g_gpt_tokenizer.decode(
        torch.argmax(sentences[:, 0, :], 1).tolist())
    print(sentence)


def print_src(sentences):
    # sentences = [sentence len, batch size]
    print("src: ", end="")

    sentence = g_bert_tokenizer.decode(sentences[:, 0].tolist())
    print(sentence)


def create_chat_tsv(filename, sentence):
    f_csv = open("datasets/" + filename + '.tsv',
                 'w',
                 encoding='utf-8',
                 newline='')
    csv_writer = csv.writer(f_csv, delimiter=delimiter)
    csv_writer.writerow([sentence, sentence])


def talk(num):
    for i in range(num):
        user_input = input("user: ")
        create_chat_tsv("chat", user_input)
        test_data = TabularDataset(path='datasets/chat.tsv',
                                   format='tsv',
                                   skip_header=False,
                                   fields=g_data_fields)
        test_iterator = BucketIterator(test_data,
                                       batch_size=opt.batch_size,
                                       sort_key=lambda x: len(x.src),
                                       sort_within_batch=False,
                                       device=g_device)
        chat(g_model, test_iterator)


def chat(model, iterator):
    model.eval()

    with torch.no_grad():
        for batch in tqdm(iterator):
            src = batch.src
            tgt = batch.tgt
            teacher_forcing_ratio = 0  # turn off teacher forcing
            output = model(src,
                           tgt,
                           response_embeds_len=tgt.size(0) - 1,
                           response_len=2 * tgt.size(0) - 1,
                           teacher_forcing_ratio=teacher_forcing_ratio)
            # tgt = [tgt len, batch size]
            # output = [tgt len, batch size, output dim]
            print_answer(output)
            # print_src(src)
    return


if TRAIN:
    train_epoch(opt)
else:
    g_model.load_state_dict(torch.load(opt.chat_model_ckpt))
    talk(5)