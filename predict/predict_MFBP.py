import torch
import numpy as np
import sys
sys.path.append('..')
from utils.data_helpers import process_input


def get_MFBP_tokenizer():
    import os
    from transformers import AutoTokenizer
    project_dir = os.path.abspath('.')

    MFBP_config_PATH = project_dir + "/MFBP_alldata/"
    tokenizer = AutoTokenizer.from_pretrained(MFBP_config_PATH, do_lower_case=False)
    return tokenizer


def get_MFBP_model(model_num):
    import os
    import torch
    from Task.TaskForMultiPeptide_MFBP import TrainConfig
    from models.BertForMultiLabelWithGAT_MFBP import BertForMultiLabelSequenceClassification

    project_dir = os.path.abspath('.')

    MFBP_config = TrainConfig(model_num)
    MFBP_config_PATH = project_dir + "/MFBP_alldata/"
    MFBP_model_PATH = project_dir  + f"/MFBP_alldata/model_Peptide_MFBP_alldata{model_num}.bin"

    finetune_model = BertForMultiLabelSequenceClassification(MFBP_config, MFBP_config.pretrained_model_dir) 

    print(MFBP_model_PATH)
    if os.path.exists(MFBP_model_PATH):
        loaded_paras = torch.load(MFBP_model_PATH)
        finetune_model.load_state_dict(loaded_paras)
        print(f"## Successfully loaded {MFBP_model_PATH} model for inference ......")
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


def pred_MFBP(seq, MFBP_model_list, MFBP_tokenizer):
    label_list = ["ACP", "ADP", "AHP", "AIP", "AMP"]
        
    # 10个模型预测取平均
    for i, model in enumerate(MFBP_model_list): 
        pred_res = get_seq_pred(seq, model, MFBP_tokenizer, 'MFBP')
        if i == 0:
            pred_res_allmodel = pred_res[0]
        else:
            pred_res_allmodel = np.vstack((pred_res_allmodel, pred_res[0]))

    end_pred_res = pred_res_allmodel.mean(axis=0)

    end_pred_res = end_pred_res > 0.5

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
        MFBP_tokenizer = get_MFBP_tokenizer()
        MFBP_model_list = list()
        for i in range(10):
            MFBP_model = get_MFBP_model(i)
            MFBP_model_list.append(MFBP_model)
        pred_MFBP(args.seq, MFBP_model_list, MFBP_tokenizer)
