import spacy
from tqdm import tqdm
import torch
from preprocess import Dataset
from torchtext.data import BucketIterator
from model import Encoder, Attention, Decoder, Seq2Seq


def translate_sentence(model, sentences, src_field, trg_field, device):
    model.to(device)
    tokenizer = spacy.load("de")
    for sentence in sentences:
        tokens = [token.text.lower() for token in tokenizer(sentence)]
        tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        src_idx = [src_field.vocab.stoi[token] for token in tokens]
        src_tensor = torch.tensor(src_idx, dtype=torch.long).unsqueeze(1).to(device)
        src_len = torch.tensor([len(src_idx)], dtype=torch.long).to(device)
        with torch.no_grad():
            encoder_outputs, hidden = model.encoder(src_tensor, src_len)

        trg_idx = [trg_field.vocab.stoi[trg_field.init_token]]
        mask = (src_tensor != model.src_pad_idx).permute(1, 0).to(device)
        for i in range(99):
            input_dec = torch.tensor(trg_idx[-1], dtype=torch.long).unsqueeze(1).to(device)

            with torch.no_grad():
                output, hidden, _ = model.decoder(input_dec, hidden, encoder_outputs, mask)

            trg_idx.append(output.argmax(1))

        result = []
        for idx in trg_idx[1:]:
            if idx.item() == trg_field.vocab.stoi[trg_field.eos_token]:
                break

            result.append(trg_field.vocab.itos[idx.item()])

        print("DE: {}\nEN: {}".format(tokens[1: -1], result))
        print("-" * 50)


def main(fpath):
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    ENC_HID_DIM = 512
    DEC_HID_DIM = 512
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    device = torch.device('cuda')
    dataset = Dataset()
    INPUT_DIM = len(dataset.SRC.vocab)
    OUTPUT_DIM = len(dataset.TRG.vocab)
    SRC_PAD_IDX = dataset.SRC.vocab.stoi[dataset.SRC.pad_token]

    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attention = Attention(ENC_HID_DIM, DEC_HID_DIM)
    decoder = Decoder(DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, OUTPUT_DIM, DEC_DROPOUT, attention)
    model = Seq2Seq(encoder, decoder, SRC_PAD_IDX, device)
    model.load_state_dict(torch.load("best_model.pt"))
    model.to(device)
    with open(fpath, "r") as f:
        sentences = f.readlines()

    translate_sentence(model, sentences, dataset.SRC, dataset.TRG, device)


if __name__ == '__main__':
    main("/data/chiendb/Data/test.txt")
