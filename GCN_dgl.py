import gc
import math
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from sklearn.metrics import f1_score

class GCN(object):
    def __init__(self, adj, adj_eval, features, labels, tvt_nids, n_layers=1, hidden_size=128, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        # config device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # fix random seeds if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.load_data(adj, adj_eval, features, labels, tvt_nids)

        self.model = GCN_model(self.features.size(1),
                               hidden_size,
                               self.n_class,
                               n_layers,
                               F.relu,
                               dropout)
        # move everything to device
        self.model.to(self.device)

    def load_data(self, adj, adj_eval, features, labels, tvt_nids):
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        if self.features.size(1) in (1433, 3703):
            self.features = F.normalize(self.features, p=1, dim=1)
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        if len(self.labels.size()) == 1:
            self.n_class = len(torch.unique(self.labels))
        else:
            self.n_class = labels.size(1)
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]
        # adj for training
        assert sp.issparse(adj)
        if not isinstance(adj, sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        adj.setdiag(1)
        self.adj = adj
        adj = sp.csr_matrix(adj)
        self.G = DGLGraph(self.adj)
        self.G = self.G.to(self.device)
        # normalization (D^{-1/2})
        degs = self.G.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.device)
        self.G.ndata['norm'] = norm.unsqueeze(1)
        # adj for inference
        assert sp.issparse(adj_eval)
        if not isinstance(adj_eval, sp.coo_matrix):
            adj_eval = sp.coo_matrix(adj_eval)
        adj_eval.setdiag(1)
        adj_eval = sp.csr_matrix(adj_eval)
        self.adj_eval = adj_eval
        self.G_eval = DGLGraph(self.adj_eval)
        self.G_eval = self.G_eval.to(self.device)
        # normalization (D^{-1/2})
        degs_eval = self.G_eval.in_degrees().float()
        norm_eval = torch.pow(degs_eval, -0.5)
        norm_eval[torch.isinf(norm_eval)] = 0
        norm_eval = norm_eval.to(self.device)
        self.G_eval.ndata['norm'] = norm_eval.unsqueeze(1)

    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # data
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()

        best_vali_acc = 0.0
        best_logits = None
        for epoch in range(self.epochs):
            self.model.train()
            logits = self.model(self.G, features)
            # losses
            l = nc_criterion(logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # validate with original graph (without dropout)
            self.model.eval()
            with torch.no_grad():
                logits_eval = self.model(self.G_eval, features).detach().cpu()
            vali_acc, _ = self.eval_node_cls(logits_eval[self.val_nid], labels[self.val_nid].cpu())
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                best_logits = logits_eval
                test_acc, conf_mat = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid].cpu())
        del self.model, features, labels, self.G
        torch.cuda.empty_cache()
        gc.collect()
        return test_acc, best_vali_acc, best_logits

    def eval_node_cls(self, logits, labels):
        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(logits))
        else:
            preds = torch.argmax(logits, dim=1)
        micro_f1 = f1_score(labels, preds, average='micro')
        return micro_f1, 1


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        if self.dropout:
            h = self.dropout(h)
        h = torch.mm(h, self.weight)
        # normalization by square root of src degree
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(fn.copy_src(src='h', out='m'),
                     fn.sum(msg='m', out='h'))
        h = g.ndata.pop('h')
        # normalization by square root of dst degree
        h = h * g.ndata['norm']
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

class GCN_model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN_model, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_feats, n_hidden, activation, 0.))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
        # output layer
        self.layers.append(GCNLayer(n_hidden, n_classes, None, dropout))

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

class CLF(object):
    def __init__(self, features, labels, tvt_nids, nums, hidden_size=128, n_layers=1, epochs=200, seed=-1, lr=1e-2, weight_decay=5e-4, dropout=0.5):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.nums = nums
        # config device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # fix random seeds if needed
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.load_data(features, labels, tvt_nids)

        self.model = CLF_Model(self.features.size(2),
                               hidden_size,
                               self.n_class,
                               n_layers,
                               F.relu,
                               dropout,
                               self.nums)
        # move everything to device
        self.model.to(self.device)

    def load_data(self, features, labels, tvt_nids):
        if isinstance(features, torch.FloatTensor):
            self.features = features
        else:
            self.features = torch.FloatTensor(features)
        if self.features.size(2) in (1433, 3703):
            self.features = F.normalize(self.features, p=1, dim=1)
        if len(labels.shape) == 2:
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        self.labels = labels
        if len(self.labels.size()) == 1:
            self.n_class = len(torch.unique(self.labels))
        else:
            self.n_class = labels.size(1)
        self.train_nid = tvt_nids[0]
        self.val_nid = tvt_nids[1]
        self.test_nid = tvt_nids[2]

    def fit(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # data
        features = self.features.to(self.device)
        labels = self.labels.to(self.device)
        # loss function for node classification
        if len(self.labels.size()) == 2:
            nc_criterion = nn.BCEWithLogitsLoss()
        else:
            nc_criterion = nn.CrossEntropyLoss()

        best_vali_acc = 0.0
        best_logits = None
        for epoch in range(self.epochs):
            self.model.train()
            logits = self.model(features)
            # losses
            l = nc_criterion(logits[self.train_nid], labels[self.train_nid])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                logits_eval = self.model(features).detach().cpu()
            vali_acc, _ = self.eval_node_cls(logits_eval[self.val_nid], labels[self.val_nid].cpu())
            if vali_acc > best_vali_acc:
                best_vali_acc = vali_acc
                best_logits = logits_eval
                test_acc, conf_mat = self.eval_node_cls(logits_eval[self.test_nid], labels[self.test_nid].cpu())
        del self.model, features, labels
        torch.cuda.empty_cache()
        gc.collect()
        return test_acc, best_vali_acc, best_logits

    def eval_node_cls(self, logits, labels):
        if len(labels.size()) == 2:
            preds = torch.round(torch.sigmoid(logits))
        else:
            preds = torch.argmax(logits, dim=1)
        micro_f1 = f1_score(labels, preds, average='micro')
        return micro_f1, 1

class CLF_Model(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 nums):
        super(CLF_Model, self).__init__()
        self.nums = nums
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.final_layer = nn.Linear(self.nums*n_hidden, n_classes)
        self.se = SELayer(channel=self.nums)

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = F.relu(layer(h))
        h = h.unsqueeze(3)
        h = self.se(h)
        h = h.view(h.size(0),-1)
        return self.final_layer(h)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)