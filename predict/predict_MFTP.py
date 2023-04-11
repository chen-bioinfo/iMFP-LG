
import torch
import numpy as np
import sys
sys.path.append('..')
from utils.data_helpers import process_input


def get_MFTP_tokenizer():
    import os
    from transformers import AutoTokenizer
    project_dir = os.path.abspath('.')
    MFTP_config_PATH = project_dir + "/MFTP_alldata/"
    tokenizer = AutoTokenizer.from_pretrained(MFTP_config_PATH, do_lower_case=False)
    return tokenizer


def get_MFTP_model(model_num):
    import os
    import torch
    from Task.TaskForMultiPeptide_MFTP import TrainConfig
    from models.BertForMultiLabelWithGAT_MFTP import BertForMultiLabelSequenceClassification

    project_dir = os.path.abspath('.')

    MFTP_config = TrainConfig(model_num)
    MFTP_config_PATH = project_dir + "/MFTP_alldata/"
    MFTP_model_PATH = project_dir  + f"/MFTP_alldata/model_Peptide_MFTP_alldata{model_num}.bin"
    
    finetune_model = BertForMultiLabelSequenceClassification(MFTP_config, MFTP_config.pretrained_model_dir) 


    if os.path.exists(MFTP_model_PATH):
        loaded_paras = torch.load(MFTP_model_PATH)
        finetune_model.load_state_dict(loaded_paras)
        print(f"## Successfully loaded {MFTP_model_PATH} model for inference ......")
    else:
        print("Model not found")

    return finetune_model


def get_seq_pred(seq, model, tokenizer, type):
    if type == 'MFBP':
        num_label = 5
    elif type == 'MFTP':
        num_label = 21
    else:
        print('type error')
        
    process_seq = process_input(seq)
    inputs = tokenizer.encode(process_seq, return_tensors='pt') 
    graph_info = np.ones((num_label,num_label))
    x = np.diag([1 for _ in range(num_label)])
    graph_info = graph_info - x
    adj_matrix = torch.from_numpy(graph_info).float()
    model.eval()
    with torch.no_grad():
        logits, _, _, _= model(
            input_ids=inputs,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            adj_matrix=adj_matrix)
    # print(logits)
    logits = logits.sigmoid().numpy()
    logits = logits

    return logits


def pred_MFTP(seq, MFTP_model_list, MFTP_tokenizer):
    label_list = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
            'AVP',
            'BBP', 'BIP',
            'CPP', 'DPPIP',
            'QSP', 'SBP', 'THP']    

    for i, model in enumerate(MFTP_model_list): 
        pred_res = get_seq_pred(seq, model, MFTP_tokenizer, 'MFTP')
        if i == 0:
            pred_res_allmodel = pred_res[0]
        else:
            pred_res_allmodel = np.vstack((pred_res_allmodel, pred_res[0]))
    end_pred_res = pred_res_allmodel.mean(axis=0)
    end_pred_res = end_pred_res > 0.5
    print(end_pred_res)

    for i, res in enumerate(end_pred_res):
        if res == True:
            print(label_list[i], end=' ')
    print()
    return end_pred_res


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str, default = None, help='Sequence to be predicted')
    args = parser.parse_args()
    if args.seq == None:
        print("Please enter the sequence to be predicted, for example: FGLPMLSILPKALCILLKRKC")
    else:
        MFTP_tokenizer = get_MFTP_tokenizer()
        MFTP_model_list = list()
        for i in range(10):
            MFTP_model = get_MFTP_model(i)
            MFTP_model_list.append(MFTP_model)
        pred_MFTP(args.seq, MFTP_model_list, MFTP_tokenizer)