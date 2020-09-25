import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, output_dim, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(enc_hid_dim * 2 + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear(enc_hid_dim * 2 + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_dec, hidden, encoder_outputs, mask):
        input_dec = input_dec.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_dec))
        weight_attn = self.attention(hidden, encoder_outputs, mask)
        weight_attn = weight_attn.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn = torch.bmm(weight_attn, encoder_outputs)
        attn = attn.permute(1, 0, 2)
        rnn_input = torch.cat((embedded, attn), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        prediction = self.fc_out(torch.cat((output.squeeze(0), attn.squeeze(0), embedded.squeeze(0)), dim=1))
        return prediction, hidden.squeeze(0), attn.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        output_dim = self.decoder.output_dim
        mask = (src != self.src_pad_idx).permute(1, 0)
        encoder_outputs, hidden = self.encoder(src, src_len)
        outputs = torch.zeros(trg_len, batch_size, output_dim).to(self.device)
        input_dec = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input_dec, hidden, encoder_outputs, mask)
            outputs[t] = output
            teacher_forcing = random.random() < teacher_forcing_ratio

            if teacher_forcing:
                input_dec = trg[t]
            else:
                input_dec = output.argmax(1)

        return outputs
