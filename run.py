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
def print_chat(sentences):
    print("chatbot: ", end="")
    for word_embeds in sentences:
        word_embed = word_embeds[0]
        # find one shot index from word embedding
        max_idx_t = word_embed.argmax()
        max_idx = max_idx_t.item()
        word = g_tokenizer.convert_ids_to_tokens(max_idx)
        print(word, end=" ")
    print("")  # new line at the end of sentence


def print_tgt(sentences):
    print("tgt: ", end="")
    for word_embeds in sentences:
        word_embed = word_embeds[0]
        max_idx = word_embed.item()
        word = g_tokenizer.convert_ids_to_tokens(max_idx)
        print(word, end=" ")
    print("")  # new line at the end of sentence


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
                                       sort_within_batch=False)
        chat(g_model, test_iterator, criterion)


def chat(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src.cuda()
            tgt = batch.tgt.cuda()

            teacher_forcing_ratio = 0  # turn off teacher forcing
            output = model(src, tgt[0:-1], teacher_forcing_ratio)
            print(f"output shape : {output.shape}")

            print_chat(output)
            # print_tgt(tgt)
            # print_tgt(output)

            # tgt = [tgt len, batch size]
            # output = [tgt len, batch size, output dim]

            # output_dim = output.shape[-1]

            # tgt = [(tgt len - 1) * batch size]
            # output = [(tgt len - 1) * batch size, output dim]
            # loss = criterion(output.view(-1, output_dim), tgt[1:].view(-1))

            # print(
            #     f'\t Val. Loss: {loss:.3f} |  Val. PPL: {math.exp(loss):7.3f}')

            # epoch_loss += loss.item()

    return


if TRAIN:
    train_epoch(opt)
else:
    g_model.load_state_dict(torch.load(opt.chat_model_ckpt))
    talk(5)