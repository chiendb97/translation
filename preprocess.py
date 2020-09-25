import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field


class Dataset:
    def __init__(self):
        self.spacy_de = spacy.load("de")
        self.spacy_en = spacy.load("en")

    def tokenize_de(self, text):
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def build_dataset(self):
        self.SRC = Field(tokenize=self.tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True,
                         include_lengths=True)
        self.TRG = Field(tokenize=self.tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True)
        train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(self.SRC, self.TRG))
        self.SRC.build_vocab(train_data, min_freq=2)
        self.TRG.build_vocab(train_data, min_freq=2)
        return train_data, valid_data, test_data
