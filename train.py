# from model import *
from model2 import *

import torch.optim as optim
import math
import time
from tqdm import tqdm
from config import Config

opt = Config()


def freeze_layers(layers):
    for layer in layers:
        for name, value in layer.named_parameters():
            value.requires_grad = False


def freeze_params(model, keywords):
    for name, param in model.named_parameters():
        for keyword in keywords:
            if name.find(keyword) != -1:
                param.requires_grad = False


def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


# freeze the bert and gpt parameters
freeze_params(g_model, ['bert.', 'gpt.'])
# freeze_layers([model.encoder.embedding, model.decoder.embedding])


def init_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            nn.init.normal_(param.data, mean=0, std=0.01)
            # nn.init.uniform_(param.data, -0.1, 0.1)


init_weights(g_model)

# print the model structure
# print(g_model)
# print_params(g_model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(g_model):,} trainable parameters')

optimizer = optim.Adam(g_model.parameters(), lr=opt.learning_rate)
criterion = nn.CrossEntropyLoss()


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    # g_bert.eval()
    # g_gpt.eval()

    print("Batch num: " + str(len(iterator)))

    epoch_loss = 0

    for batch in tqdm(iterator):
        src = batch.src.cuda()
        tgt = batch.tgt.cuda()
        optimizer.zero_grad()
        teacher_forcing_ratio = 1
        output = model(src, tgt[:-1], teacher_forcing_ratio)
        # tgt = [tgt len, batch size]
        # output = [tgt len, batch size, output dim]
        output_dim = output.shape[-1]

        # tgt = [(tgt len - 1) * batch size]
        # output = [(tgt len - 1) * batch size, output dim]
        loss = criterion(output[:-1].view(-1, output_dim), tgt[1:-1].view(-1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        print(f"Batch loss: {loss.item()}")

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.tgt
            teacher_forcing_ratio = 0  # turn off teacher forcing
            output = model(src,
                           tgt,
                           response_embeds_len=tgt.size(0) - 1,
                           response_len=tgt.size(0) - 1,
                           teacher_forcing_ratio=teacher_forcing_ratio)
            # tgt = [tgt len, batch size]
            # output = [tgt len, batch size, output dim]

            output_dim = output.shape[-1]
            loss = criterion(output.view(-1, output_dim), tgt[1:].view(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_epoch(opt):
    # best_validation_loss = float('inf')
    start_epoch = 1
    if opt.continue_train:
        g_model.load_state_dict(torch.load(opt.chat_model_ckpt))
        try:
            start_epoch += (int)(opt.chat_model_ckpt[-2:])
        except:
            start_epoch += (int)(opt.chat_model_ckpt[-1:])
        print(f"Loading checkpoints from epoch {start_epoch - 1} successes!")
    for epoch in range(opt.epoches):
        start_time = time.time()

        train_loss = train(g_model, train_iterator, optimizer, criterion, CLIP)
        # validation_loss = evaluate(model, validation_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # if validation_loss < best_validation_loss:
        #     best_validation_loss = validation_loss
        #     torch.save(model.state_dict(), 'chatbot_rnn-model.pt')

        print(
            f'\nEpoch: {epoch + start_epoch} | Time: {epoch_mins}m {epoch_secs}s'
        )
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}'
        )
        #print(f'\t Val. Loss: {validation_loss:.3f} |  Val. PPL: {math.exp(validation_loss):7.3f}')

        if epoch % opt.save_per_epoch == 0:
            checkpoints_path = 'checkpoints/checkpoints_{time}_epoch{epoch}'.format(
                time=time.strftime('%m%d_%H%M'), epoch=start_epoch + epoch)
            torch.save(g_model.state_dict(), checkpoints_path)  # ??????????????????
