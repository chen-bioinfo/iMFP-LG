# -*- coding: utf-8 -*-
# @Author  : twd
# @FileName: demo.py
# @Software: PyCharm

"""
In order to ensure that the divided training data and test data are consistent with the original paper
(Identifying multi-functional bioactive peptide functions using multi-label deep learning),
the data processing code comes from the original paper code https://github.com/xialab-ahu/MLBP/blob/master/MLBP/main.py
"""
import os
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split



def GetSourceData(root, dir, lb):
    seqs = []
    print('\n')
    print('now is ', dir)
    file = '{}CD_.txt'.format(dir)
    file_path = os.path.join(root, dir, file)

    with open(file_path) as f:
        for each in f:
            if each == '\n' or each[0] == '>':
                continue
            else:
                seqs.append(each.rstrip())

    # data and label
    label = len(seqs) * [lb]
    seqs_train, seqs_test, label_train, label_test = train_test_split(seqs, label, test_size=0.2, random_state=0)
    print('train data:', len(seqs_train))
    print('test data:', len(seqs_test))
    print('train label:', len(label_train))
    print('test_label:', len(label_test))
    print('total numbel:', len(seqs_train)+len(seqs_test))

    return seqs_train, seqs_test, label_train, label_test



def DataClean(data):
    max_len = 0
    for i in range(len(data)):
        st = data[i]
        # get the maximum length of all the sequences
        if(len(st) > max_len): max_len = len(st)

    return data, max_len

def GetSequenceData(dirs, root):
    # getting training data and test data
    count, max_length = 0, 0
    tr_data, te_data, tr_label, te_label = [], [], [], []
    for dir in dirs:
        # 1.getting data from file
        tr_x, te_x, tr_y, te_y = GetSourceData(root, dir, count)
        count += 1

        # 2.getting the maximum length of all sequences
        tr_x, len_tr = DataClean(tr_x)
        te_x, len_te = DataClean(te_x)
        if len_tr > max_length: max_length = len_tr
        if len_te > max_length: max_length = len_te
        

        # 3.dataset
        tr_data += tr_x
        te_data += te_x
        tr_label += tr_y
        te_label += te_y

    data_dir = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
    # filepath
    train_seq_path = os.path.join(data_dir, 'MFBP', 'seq_data', 'tr_seq.npy')   # MFBP
    test_seq_path = os.path.join(data_dir, 'MFBP', 'seq_data', 'te_seq.npy')
    np.save(train_seq_path, tr_data)
    np.save(test_seq_path, te_data)
    
    train_label = np.array(tr_label)
    test_label = np.array(te_label)
    return [tr_data, te_data, train_label, test_label]
    # return [train_data, test_data, train_label, test_label]

def GetData(path):
    dirs = ['AMP', 'ACP', 'ADP', 'AHP', 'AIP'] # functional peptides

    # get sequence data
    sequence_data = GetSequenceData(dirs, path)

    return sequence_data


def main():
    # I.get sequence data
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
    data_path = os.path.join(data_dir, 'MFBP', 'raw_dataset') 
    sequence_data = GetData(data_path)


if __name__ == '__main__':
    # executing the main function
    main()