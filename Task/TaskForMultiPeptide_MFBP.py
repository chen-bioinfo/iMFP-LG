import sys

sys.path.append('../')
from utils.data_helpers import LoadMultiPeptideClassificationDataset, get_json_file, process_input
from models.BertForMultiLabelWithGAT_MFBP import BertForMultiLabelSequenceClassification
from utils.log_helper import logger_init
from utils.evaluation import evaluate
from transformers import BertTokenizer
from transformers import AdamW
from models.FGM import FGM
import numpy as np
import logging
import torch
import os
import time

class TrainConfig:
    def __init__(self, model_num):
        # file path
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'data','MFBP')   # MLBP

        self.pretrained_model_dir = os.path.join(self.project_dir, 'pretrain_model', "tapebert") 

        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.train_file_path = os.path.join(self.dataset_dir, 'train_data.npy')
        self.test_file_path = os.path.join(self.dataset_dir, 'test_data.npy')
        self.data_name = 'Peptide_MFBP'

        # log path
        self.model_save_dir = os.path.join(self.project_dir, 'cache', 'MFBP')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs', 'MFBP')
        
        # Hyperparameters
        self.is_sample_shuffle = True 
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.pad_token_id = 0
        self.batch_size = 32  
        self.max_sen_len = None
        self.num_labels = 5 
        self.learning_rate = 5e-5
        self.epochs = 100
        self.model_num = model_num
        self.pooling = 'pooler'    
        self.res_path = os.path.join(self.project_dir, 'res', 'MFBP', 'result') # the save path of predicted results
        # model save path/logging path
        self.model_save_path = os.path.join(self.model_save_dir, f'model_{self.data_name}{self.model_num}.bin')
        logger_init(log_file_name=f"{self.data_name}{self.model_num}", log_level=logging.INFO, log_dir=self.logs_save_dir)
        
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        if not os.path.exists(self.logs_save_dir):
            os.makedirs(self.logs_save_dir)

        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = get_json_file(bert_config_path)
        for key, value in bert_config.items():
            self.__dict__[key] = value

        logging.info(" ### Print the current configuration to a log file ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")


def train(config):
    model = BertForMultiLabelSequenceClassification(config, config.pretrained_model_dir)    
    logging.info(f"{model}")

    model = model.to(config.device)
    model.train()
    fgm = FGM(model)

    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if 'bert' in n],
            "lr": config.learning_rate,
        },
        {
            "params": [p for n, p in model.named_parameters() if 'gat' in n],
            "lr": config.learning_rate*20,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadMultiPeptideClassificationDataset(vocab_path=config.vocab_path,
                                                          tokenizer=bert_tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter = data_loader.load_train_test_data(config.train_file_path, config.test_file_path)

    # Construct a complete graph with edge weight 1
    graph_info = np.ones((config.num_labels,config.num_labels))
    x = np.diag([1 for i in range(config.num_labels)])
    graph_info = graph_info - x
    adj_matrix = torch.from_numpy(graph_info).float().to(config.device)
    print(adj_matrix.shape)

    for epoch in range(config.epochs):
        model.train()
        losses = 0
        start_time = time.time()
        seq_labelemb = dict()   # Save the sequence embedding in the final model  {'seq':(label,embedding),..}
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.transpose(0, 1).to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            padding_mask = (sample != data_loader.PAD_IDX)
            loss, logits, embedding, _, _ = model(
                input_ids=sample,
                attention_mask=padding_mask,
                token_type_ids=None,
                position_ids=None,
                labels=label,
                adj_matrix=adj_matrix)

            optimizer.zero_grad()
            loss.backward()
            
            # adversarial perturbations  
            fgm.attack()
            loss_adv, logits, embedding, _, _ = model(
                input_ids=sample,
                attention_mask=padding_mask,
                token_type_ids=None,
                position_ids=None,
                labels=label,
                adj_matrix=adj_matrix)
            loss_adv.backward()
            fgm.restore()

            optimizer.step()

            losses += loss.item()  
            y_pred = logits.sigmoid()    
    
            if idx % 50 == 0:
                logging.info(f"Epoch: [{epoch+1}/{config.epochs}], Batch[{idx}/{len(train_iter)}], "
                             f"Train loss: {loss.item():.3f}")

        end_time = time.time()
        train_loss = losses / len(train_iter)
        if (epoch+1) % 5 == 0:
            aiming, _, _, _, _ = TestModel(train_iter, adj_matrix, model, config.device, data_loader.PAD_IDX, config.res_path, config.model_num)
            logging.info(f"Epoch: {epoch+1}, Train loss: {train_loss:.3f}, Train precision: {aiming:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        else:
            logging.info(f"Epoch: {epoch+1}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        
        # save the final model
        torch.save(model.state_dict(), config.model_save_path)

    
    # test model
    aiming, coverage, accuracy, absolute_true, absolute_false = TestModel(test_iter, adj_matrix, model, config.device, data_loader.PAD_IDX, config.res_path, config.model_num)
    logging.info(f"precision on val {aiming:.3f}")
    logging.info(f"coverage on val {coverage:.3f}")
    logging.info(f"accuracy on val {accuracy:.3f}")
    logging.info(f"absolute_true on val {absolute_true:.3f}")
    logging.info(f"absolute_false on val {absolute_false:.3f}")
    print(f"{config.data_name}_seed{config.model_num}_lr{config.learning_rate}_E{config.epochs}_BS{config.batch_size}")
        

def TestModel(data_iter, adj_matrix, model, device, PAD_IDX, res_path, model_num):
    import numpy as np
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        real_res = []
        pred_res = []
        for id, (x, y) in enumerate(data_iter):
            x, y = x.transpose(0,1).to(device), y.to(device)
            padding_mask = (x != PAD_IDX)
            
            logits, _, _, _ = model(
                input_ids=x,
                attention_mask=padding_mask,
                adj_matrix=adj_matrix)

            y_pred = logits.sigmoid()
            y_pred = y_pred.detach().cpu().numpy()
            label_ids = y.to('cpu').numpy()
            if id == 0:
                pred_res = y_pred
                real_res = label_ids
            else:
                pred_res = np.vstack((y_pred, pred_res))
                real_res = np.vstack((label_ids, real_res))

        aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(pred_res>0.5, real_res)

        model.train()
        return aiming, coverage, accuracy, absolute_true, absolute_false


def inference(config, inference_model_path, infer_seq):
    import numpy as np
    label2id = {"ACP":0, "ADP":1, "AHP":2, "AIP":3, "AMP":4}
    id2label = {0:"ACP", 1:"ADP", 2:"AHP", 3:"AIP", 4:"AMP"}
    model = BertForMultiLabelSequenceClassification(config,
                                        config.pretrained_model_dir) 

    if os.path.exists(inference_model_path):
        loaded_paras = torch.load(inference_model_path)
        model.load_state_dict(loaded_paras)
        logging.info(f"## Successfully loaded the existing model in {inference_model_path} for prediction......")
    model = model.to(config.device)
    
    model.eval()
    with torch.no_grad():
        seq = process_input(infer_seq)
        bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir, do_lower_case=False)
        tmp = bert_tokenize(seq, return_tensors='pt')
        seq_input = tmp['input_ids'].to(config.device)
        logits, _, _ = model(input_ids=seq_input)
        y_pred = (logits.sigmoid() > 0.5).reshape(-1).cpu().numpy()
        for i in range(len(y_pred)):
            if y_pred[i] == True:
                print(id2label[i], end="\t")
        print()

if __name__ == '__main__':
    for model_num in range(10):
        train_config = TrainConfig(model_num)
        train(train_config)
        

