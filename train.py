import sys
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from preprocess import Dataset
from torchtext.data import BucketIterator
from model import Encoder, Attention, Decoder, Seq2Seq


def init_weight(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            torch.nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            torch.nn.init.constant_(param.data, 0)


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    sum_loss = 0
    tqdm_iterator = tqdm(iterator)
    for data in tqdm_iterator:
        src, src_len = data.src
        trg = data.trg
        optimizer.zero_grad()
        outputs = model(src, src_len, trg)
        trg = trg[1:].view(-1)
        outputs = outputs[1:].view(-1, outputs.shape[-1])
        loss = criterion(outputs, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        sum_loss += loss.item()
        tqdm_iterator.set_description("Loss: {}".format(loss.item()))

    return sum_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    sum_loss = 0
    with torch.no_grad():
        for data in tqdm(iterator):
            src, src_len = data.src
            trg = data.trg
            outputs = model(src, src_len, trg)
            trg = trg[1:].view(-1)
            outputs = outputs[1:].view(-1, outputs.shape[-1])
            loss = criterion(outputs, trg)
            sum_loss += loss.item()

    return sum_loss / len(iterator)


def main():
    BATCH_SIZE = 32
    NUM_EPOCH = 12
    LR = 0.001
    CLIP = 1
    STEP_SIZE = 4
    GAMMA = 0.1
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    device = torch.device('cuda')

    dataset = Dataset()
    train_data, valid_data, test_data = dataset.build_dataset()
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device
    )

    INPUT_DIM = len(dataset.SRC.vocab)
    OUTPUT_DIM = len(dataset.TRG.vocab)
    SRC_PAD_IDX = dataset.SRC.vocab.stoi[dataset.SRC.pad_token]
    TRG_PAD_IDX = dataset.TRG.vocab.stoi[dataset.TRG.pad_token]

    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attention = Attention(ENC_HID_DIM, DEC_HID_DIM)
    decoder = Decoder(DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, OUTPUT_DIM, DEC_DROPOUT, attention)
    model = Seq2Seq(encoder, decoder, SRC_PAD_IDX, device)
    model.apply(init_weight)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss(ignore_index=TRG_PAD_IDX).to(device)
    scheduler = StepLR(optimizer, STEP_SIZE, GAMMA)
    
    min_valid_loss = 1e10

    for e in range(NUM_EPOCH):
        print("Epoch: {}".format(e + 1))
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        print("Train loss: {}".format(train_loss))
        valid_loss = evaluate(model, valid_iterator, criterion)
        print("Valid loss: {}".format(valid_loss))
        # scheduler.step(e)

        if valid_loss < min_valid_loss:
            torch.save(model.state_dict(), "best_model.pt")
            min_valid_loss = valid_loss

if __name__ == '__main__':
    main()
