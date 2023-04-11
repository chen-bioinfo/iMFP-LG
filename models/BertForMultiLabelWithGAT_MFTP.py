from multiprocessing import pool
from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

def adjConcat(a, b):
    """
    Combine the two matrices a,b diagonally along the diagonal direction and fill the empty space with zeros
    """
    lena = len(a)
    lenb = len(b)
    left = np.row_stack((a, np.zeros((lenb, lena))))  
    right = np.row_stack((np.zeros((lena, lenb)), b))  
    result = np.hstack((left, right)) 
    return result

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions1 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions1):
            self.add_module('attention1_{}'.format(i), attention)
        self.attentions2 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions2):
            self.add_module('attention2_{}'.format(i), attention)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        attention_matrix_list = list()   # Lists for saving attention: len(attention_matrix_list) = layer_num * head
        # First layer gat
        for idx, att in enumerate(self.attentions1):
            x1, attention_matrix = att(x, adj)
            attention_matrix_list.append(attention_matrix)
            if idx == 0:
                x_tmp = x1
            else:
                x_tmp = torch.cat((x_tmp, x1), dim=1)
        x = F.dropout(x_tmp, self.dropout, training=self.training)
        x = F.elu(x)

        # Second layer gat
        for idx, att in enumerate(self.attentions2):
            x2, attention_matrix = att(x, adj)
            attention_matrix_list.append(attention_matrix)
            if idx == 0:
                x_tmp = x2
            else:
                x_tmp = torch.cat((x_tmp, x2), dim=1)
        x = F.dropout(x_tmp, self.dropout, training=self.training)
        x = F.elu(x)

        return x, attention_matrix_list


class BertForMultiLabelSequenceClassification(nn.Module):
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForMultiLabelSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained(bert_pretrained_model_dir, output_attentions=True)
        self.config = config
        self.num_labels = config.num_labels
        self.gat = GAT(nfeat=768, nhid=128, dropout=0.2, alpha=0.2,nheads=6)   # dropout=0.2

        self.pooling = config.pooling
        for i in range(self.num_labels):    
            setattr(self, "FC%d" %i, nn.Sequential(
                                      nn.Linear(in_features=768,out_features=768),
                                      nn.Dropout()))

        for i in range(self.num_labels):    
            setattr(self, "CLSFC%d" %i, nn.Sequential(
                                      nn.Linear(in_features=768,out_features=1),
                                      nn.Dropout(),
                                      )) 

    def forward(self,
                input_ids,  # [batch_size, src_len]
                attention_mask=None,  # [batch_size, src_len]
                token_type_ids=None,  # [batch_size, src_len]
                position_ids=None,
                labels=None,  # [batch_size, src_len]
                adj_matrix = None, # [num_labels, num_labels]
                ):
        
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)
        attention_matrix = output[-1]  # [12, 1, 12, n, n]
        if self.pooling == 'cls':
            pred_emb = output.last_hidden_state[:, 0]  # [batch, 768]
        if self.pooling == 'pooler':
            pred_emb = output.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = output.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            pred_emb = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]        
        
        outs = []

        for i in range(self.num_labels):
            FClayer = getattr(self, "FC%d" %i)
            y = FClayer(pred_emb)
            y = torch.squeeze(y, dim=-1)
            outs.append(y)
        
        outs = torch.stack(outs, dim=0).transpose(0, 1)  # [batch, num_labels, 768]
        outs = outs.reshape(-1, 768)  

        for i in range(pred_emb.shape[0]):
            if i == 0:
                end_adj_matrix = adj_matrix.cpu().numpy()
            else:
                end_adj_matrix = adjConcat(end_adj_matrix, adj_matrix.cpu().numpy())    # [batch_size x num_label, batch_size x num_label]

        end_adj_matrix = torch.tensor(end_adj_matrix).to(outs.device)
        # print(end_adj_matrix.shape)
        gat_embedding, gat_attention = self.gat(outs, end_adj_matrix)
        gat_embedding = gat_embedding.reshape(-1, self.num_labels, 768)
    
        prediction_scores = list()
        for i in range(self.num_labels):
            CLSFClayer = getattr(self, "CLSFC%d" %i)
            y = CLSFClayer(gat_embedding[:,i,:])
            prediction_scores.append(y)

        prediction_res = torch.stack(prediction_scores, dim=1).reshape(-1,self.config.num_labels)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()  # weight=class_weight
            loss = loss_fct(prediction_res.view(-1, self.config.num_labels), labels.view(-1, self.config.num_labels))
            return loss, prediction_res, pred_emb, attention_matrix, gat_attention
        else:
            return prediction_res, pred_emb, attention_matrix, gat_attention # [src_len, batch_size, num_labels]

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True