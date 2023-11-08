import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution, Fastformer

class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class FastRx_wo_GCN(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=256, device=torch.device('cpu:0')):
        super(FastRx_wo_GCN, self).__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.emb_dim_ff = 128
        self.fastformer = Fastformer(dim = 2*self.emb_dim_ff, decode_dim = self.emb_dim)

        self.dropout = nn.Dropout(p=0.2)

        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, self.emb_dim_ff)

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.cnn1d = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding='same', stride=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )

    def forward(self, input):

	    # patient health representation
        i1_seq, i2_seq = [], []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))
            i2 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))

            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        i1_seq = self.cnn1d(i1_seq.permute(1, 0, 2))
        i2_seq = self.cnn1d(i2_seq.permute(1, 0, 2))
        i1_seq = i1_seq.permute(1, 0, 2)
        i2_seq = i2_seq.permute(1, 0, 2)

        h = torch.cat([i1_seq, i2_seq], dim=-1) # (seq, dim*2)

        mask = torch.ones(1, self.emb_dim).to(torch.bool).to(self.device)
        feat = self.fastformer(h, mask).squeeze(0)

        result = self.output(feat[-1:])

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

class FastRx_wo_Diag(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=256, device=torch.device('cpu:0')):
        super(FastRx_wo_Diag, self).__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.emb_dim_ff = 128
        self.fastformer = Fastformer(dim = self.emb_dim_ff, decode_dim = self.emb_dim)

        self.dropout = nn.Dropout(p=0.2)
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, self.emb_dim_ff)

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.cnn1d = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding='same', stride=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

    def forward(self, input):

	    # patient health representation
        i2_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i2 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i2_seq.append(i2)

        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)
        i2_seq = self.cnn1d(i2_seq.permute(1, 0, 2))
        i2_seq = i2_seq.permute(1, 0, 2)

        h = torch.cat([i2_seq], dim=-1) # (seq, dim*2)

        mask = torch.ones(1, self.emb_dim).to(torch.bool).to(self.device)
        feat = self.fastformer(h, mask).squeeze(0)

        # graph memory module
        '''I:generate current input'''
        query = feat[-1:] # (1,dim)
        '''G:generate graph memory bank and insert history information'''
        drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)

        if len(input) > 1:
            history_keys = feat[:(feat.size(0)-1)] # (seq-1, dim)
            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        # print(query.shape, drug_memory.t().shape)
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        result = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

class FastRx_wo_Proc(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=256, device=torch.device('cpu:0')):
        super(FastRx_wo_Proc, self).__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.emb_dim_ff = 128
        self.fastformer = Fastformer(dim = self.emb_dim_ff, decode_dim = self.emb_dim)

        self.dropout = nn.Dropout(p=0.2)
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, self.emb_dim_ff)

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.cnn1d = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding='same', stride=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.output = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

    def forward(self, input):

	    # patient health representation
        i1_seq = []

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)

        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)

        i1_seq = self.cnn1d(i1_seq.permute(1, 0, 2))
        i1_seq = i1_seq.permute(1, 0, 2)

        h = torch.cat([i1_seq], dim=-1) # (seq, dim*2)

        mask = torch.ones(1, self.emb_dim).to(torch.bool).to(self.device)
        feat = self.fastformer(h, mask).squeeze(0)

        # graph memory module
        '''I:generate current input'''
        query = feat[-1:] # (1,dim)
        '''G:generate graph memory bank and insert history information'''
        drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)

        if len(input) > 1:
            history_keys = feat[:(feat.size(0)-1)] # (seq-1, dim)
            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        result = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

class FastRx_wo_CNN1D(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, emb_dim=256, device=torch.device('cpu:0')):
        super(FastRx_wo_CNN1D, self).__init__()

        self.device = device
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        self.emb_dim_ff = 128
        self.fastformer = Fastformer(dim = 2*self.emb_dim_ff, decode_dim = self.emb_dim)

        self.dropout = nn.Dropout(p=0.2)
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(voc_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.embedding = nn.Embedding(vocab_size[0] + vocab_size[1] + 2, self.emb_dim_ff)

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

        self.output = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )

    def forward(self, input):

	    # patient health representation
        i1_seq = []
        i2_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))))
            i2 = mean_embedding(self.dropout(self.embedding(torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))

            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)

        h = torch.cat([i1_seq, i2_seq], dim=-1) # (seq, dim*2)

        mask = torch.ones(1, self.emb_dim).to(torch.bool).to(self.device)
        feat = self.fastformer(h, mask).squeeze(0)

        # graph memory module
        '''I:generate current input'''
        query = feat[-1:] # (1,dim)
        '''G:generate graph memory bank and insert history information'''
        drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)

        if len(input) > 1:
            history_keys = feat[:(feat.size(0)-1)] # (seq-1, dim)
            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device) # (seq-1, size)

        '''O:read from global memory bank and dynamic memory bank'''
        # print(query.shape, drug_memory.t().shape)
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t())) # (1, seq-1)
            weighted_values = visit_weight.mm(history_values) # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory) # (1, dim)
        else:
            fact2 = fact1
        '''R:convert O and predict'''
        result = self.output(torch.cat([query, fact1, fact2], dim=-1)) # (1, dim)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg