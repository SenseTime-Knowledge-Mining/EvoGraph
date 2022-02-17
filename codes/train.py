import os
import json
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
import torch
from edge_manipulation import Edge_Manipulation

import copy



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='blogcatalog')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()


    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0


    tvt_nids = pickle.load(open(f'data/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{args.dataset}_labels.pkl', 'rb'))
    if sp.issparse(features):
        features = torch.FloatTensor(features.toarray())

    if args.dataset == 'blogcatalog':
        add_e1, remove_e1, add_e2, remove_e2 = 0.001,0.875*171743*2/(5196*5195),0.001,0.875*171743*2/(5196*5195)
    elif args.dataset == 'flickr':
        add_e1, remove_e1, add_e2, remove_e2 = 0.001, 0.008 , 0.001, 0.008


    nums = 3
    acc = [[] for _ in range(nums+1)]
    for _i in range(30):
        model = Edge_Manipulation(args.dataset, adj_orig, features, labels, tvt_nids, args.gnn, nums, add_e1, remove_e1, add_e2, remove_e2)
        temp_res = model.fit()
        for _j in range(nums+1):
            acc[_j].append(temp_res[_j])
        #print(temp_res)
    for _i in range(nums + 1):
        print(f'Micro F1: {np.mean(acc[_i]):.6f}, std: {np.std(acc[_i]):.6f}')
