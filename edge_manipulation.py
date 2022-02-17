import os
import copy
import torch
import numpy as np
import scipy.sparse as sp
from GCN_dgl import GCN, CLF
from GAT_dgl import GAT
from GSAGE_dgl import GraphSAGE
from sklearn.metrics import f1_score


class Edge_Manipulation(object):
    def __init__(self, dataset, adj_matrix, features, labels, tvt_nids, gnn, nums, add_e1, remove_e1, add_e2, remove_e2, max_t=5):
        self.dataset = dataset
        # hyperparameters
        self.nums = nums
        self.max_t = max_t
        self.gnn = gnn
        self.add_e = [add_e1,add_e2]
        self.remove_e = [remove_e1,remove_e2]
        # data
        self.adj = adj_matrix
        self.features = features
        self.labels = labels
        self.tvt_nids = tvt_nids


    def fit(self):
        if self.gnn == 'gcn':
            GNN = GCN
        elif self.gnn == 'gat':
            GNN = GAT
        elif self.gnn == 'gsage':
            GNN = GraphSAGE
        adj_temp = self.adj
        f1_list, logits_list = [], []
        for i in range(self.nums):
            model = GNN(adj_temp, adj_temp, self.features, self.labels, self.tvt_nids)
            test_f1, val_f1, logits1 = model.fit()
            f1_list.append(test_f1)
            logits_list.append(logits1)
            if i != self.nums-1:
                adj_temp = self.adjustGraph(adj_temp, logits1,self.add_e[i],self.remove_e[i])

        new_logits_list = [i.unsqueeze(1) for i in logits_list]
        logi = torch.cat(new_logits_list,1)
        model = CLF(logi, self.labels, self.tvt_nids,self.nums)
        test_f1_final, val_f1_final, logits = model.fit()
        f1_list.append(test_f1_final)

        return f1_list

    def eval_f1(self, logits, labels):
        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(logits))
        else:
            preds = torch.argmax(logits, dim=1)
        micro_f1 = f1_score(labels, preds, average='micro')
        return micro_f1, 1

    def sim(self, a, b):
        res = np.linalg.norm(a-b)
        return res

    def adjustGraph(self,adj, logits,add_e0,remove_e0):
        adj = copy.deepcopy(adj)
        adj = sp.lil_matrix(adj)
        adj_new = adj
        logi = logits.numpy()

        total = adj_new.shape[0] * (adj_new.shape[0] - 1) / 2.0
        add_num = int(total * add_e0)
        remove_num = int(total * remove_e0)


        dic1, dic2 = {}, {}
        for i in range(adj_new.shape[0]):
            for j in range(i+1, adj_new.shape[0]):
                if adj_new[i, j] == 0:
                    dic1[str(i)+':'+str(j)] = self.sim(logi[i],logi[j])
                if adj_new[i, j] == 1:
                    dic2[str(i)+':'+str(j)] = self.sim(logi[i], logi[j])
        dic1_order = sorted(dic1.items(), key=lambda x: x[1], reverse=False)
        dic2_order = sorted(dic2.items(), key=lambda x: x[1], reverse=True)

        add_cnt, remove_cnt = 0, 0
        for k in dic1_order:
            if add_num==0:
                break
            adj_new[int(k[0].split(':')[0]),int(k[0].split(':')[1])] = 1
            adj_new[int(k[0].split(':')[1]),int(k[0].split(':')[0])] = 1
            add_num = add_num-1
            add_cnt+=1
        for k in dic2_order:
            if remove_num==0:
                break
            adj_new[int(k[0].split(':')[0]),int(k[0].split(':')[1])] = 0
            adj_new[int(k[0].split(':')[1]),int(k[0].split(':')[0])] = 0
            remove_num = remove_num-1
            remove_cnt+=1

        return sp.csr_matrix(adj_new)
