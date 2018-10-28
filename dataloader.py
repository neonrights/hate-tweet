import torch
import pandas as pd

from torch.utils import data
from nltk.tokenize import TweetTokenizer
from elmoformanylangs import Embedder

def build_dataset(self, data_path, max_seq_len=None):
    dataframe = pd.read_csv(data_path, sep='\t')
    dataframe = dataframe.set_index('id')

    self.vocab = dict()
    self.labels = dataframe[['HS', 'TR', 'AG']].as_matrix()
    for text in dataframe['text']:
        pass # split and build vocabulary


class ELMoDataset(data.Dataset):
    def __init__(self, data_path, max_seq_len=None, lang='en'):
        df = pd.read_csv(data_path, sep='\t')
        df = df.set_index('id')
        
        tokenizer = TweetTokenizer()
        
        self.labels = df[['HS', 'TR', 'AG']].as_matrix()
        self.text = [tokenizer.tokenize(text) for text in df['text']]
        self.max_seq_len = max_seq_len

        if lang == 'en':
            self.elmo = Embedder('elmo/english/')
        else lang == 'es':
            self.elmo = Embedder('elmo/spanish/')

        self.text = self.elmo.sent2elmo(self.text)

    def __len__(self):
        return len(self.text)

    def __getitem(self, index):
        if self.max_seq_len is not None:
            return self.text[index], self.labels[index]
        else: # pad or trim sequence as necessary
            return self.text[index], self.labels[index]
        


class Dataset(data.Dataset):
    def __init__(self, cleaned_data):
        self.data = cleaned_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc(index)


