import torch
from torch.utils.data import DataLoader
import json
import logging
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import six



def get_json_file(json_file):
    """Constructs a `BertConfig` from a json file of parameters.
       Read configuration information from json configuration file"""
    dict = {}
    with open(json_file, 'r') as reader:
        text = reader.read()
    json_file = json.loads(text)
    for (key, value) in six.iteritems(json_file):
        dict[key] = value
    return dict


class Vocab:
    """
    Construct a vocabulary based on the local vocab file
    vocab = Vocab()
    print(vocab.itos)  # Get a list, returning each amino acid(AA) in the vocabulary;
    print(vocab.itos[2])  # Return the AAs in the vocabulary by index;
    print(vocab.stoi)  # Get a dictionary and return the index of each AAs in the vocabulary;
    print(vocab.stoi['A'])  # Get the index in the vocabulary by the AAs
    print(len(vocab))  # Returns the length of the vocabulary
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    return Vocab(vocab_path)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: whether to put batch_size in the first dimension
        padding_value:
        max_len :
                When max_len = 50, it means that the sample is padding with a fixed length, and the excess is truncated;
                When max_len=None, it means padding others with the length of the longest sample in the current batch;
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


def process_input(seq):
    """
    Separate the input sequence with spaces   (eg. "ABCD"--->"A B C D")
    param: seq: input sequence
    return: Sequence separated by spaces
    """
    pro_seq = ''
    for i in range(len(seq)):
        if i == 0:
            pro_seq += seq[i]
        else:
            pro_seq += " " + seq[i]
    return pro_seq


class LoadMultiPeptideClassificationDataset:
    def __init__(self,
                 vocab_path='./vocab.txt', 
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True
                 ):
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']

        self.batch_size = batch_size
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle


    def data_process(self, filepath):
        """
        Convert each amino acid in each sequence into the form of an index according to the dictionary, 
        and return the length of the longest sample among all samples
        :param filepath: dataset path
        :return:
        """
        seq_label_dict = np.load(filepath, allow_pickle=True).item()
        print(len(seq_label_dict.keys()))
        data = list()
        max_len = 0
        for seq in seq_label_dict.keys():
            s = seq

            # process sequences
            s = process_input(s)
            s = s.split(' ')      # Change to ['G', 'L', 'F', 'D', 'I'] format
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in s] + [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)

            # process labels
            l = seq_label_dict[seq]
            l = l.astype(float) 
            l = torch.tensor(l, dtype=torch.float)
            
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))
        return data, max_len

    def load_train_test_data(self, train_file_path=None,
                                 test_file_path=None,
                                 only_test=False):
        test_data, _ = self.data_process(filepath=test_file_path)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process(filepath=train_file_path)
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        return train_iter, test_iter

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch: # process each sample in a batch
            batch_sentence.append(sen)
            batch_label.append(label.tolist())
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.float)
        return batch_sentence, batch_label
