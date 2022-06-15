import torch
import numpy as np
from . import Utils
from .Dataset import Dataset

class Graph:
    def __init__(self, adj, labels, features, idx_train, idx_val, idx_test, split_seed, device):
        self.adj = adj.to(device)
        self.features = features.to(device)
        self.labels = labels.to(device)
        self.idx_train = idx_train.to(device)
        self.idx_val = idx_val.to(device)
        self.idx_test = idx_test.to(device)
        self.split_seed = split_seed

    def __repr__(self):
        return f"<Graph {self.adj.shape[0]}x{self.adj.shape[1]}>"

    def summarize(self, name=""):
        print()
        print(f'[i] Dataset Summary: {name}')
        print(f'\tadj shape: {list(self.adj.shape)}')
        print(f'\tfeature shape: {list(self.features.shape)}')
        print(f'\tnum labels: {self.labels.max().item()+1}')
        print(f'\tsplit seed: {self.split_seed}')
        print(
            f'\ttrain|val|test: {self.idx_train.sum()}|{self.idx_val.sum()}|{self.idx_test.sum()}')
    
    def split(self, nsplits):
        indices = torch.zeros(10, dtype=torch.bool)
        return None

    def numEdges(self):
        return self.adj.sum() / 2

    def numNodes(self):
        return self.adj.shape[0]

    def getSample(self, size):
        indices = (torch.bernoulli(torch.empty(1, size)[0].uniform_(0,1))) > 0.5
        maskA = indices.nonzero().t()[0]
        maskB = (~indices).nonzero().t()[0]

        return maskA, maskB

    def getSubgraph(self, indices):
        return Graph(
            adj=self.adj[indices].t()[indices].t(),
            features=self.features[indices],
            labels=self.labels[indices],
            idx_train=self.idx_train[indices],
            idx_val=self.idx_val[indices],
            idx_test=self.idx_test[indices],
            device=self.device
        )

    def getEntropies(self):
        # TODO
        return 0

def getGraph(root, name, setting, seed, device, verbose=True):
    data = Dataset(root, name, setting, seed)

    adj = torch.LongTensor(data.adj.todense())
    adj = Utils.make_symmetric(adj)
    labels = torch.LongTensor(data.labels)
    features = torch.FloatTensor(np.array(data.features.todense()))

    def indices_to_bool(indices, length):
        arr = torch.zeros(length)
        arr[indices] = 1
        return arr > 0

    idx_train = Utils.idx_to_bool(data.idx_train, features.shape[0])
    idx_val = Utils.idx_to_bool(data.idx_val, features.shape[0])
    idx_test = Utils.idx_to_bool(data.idx_test, features.shape[0])
    
    return Graph(adj, labels, features, idx_train, idx_val, idx_test, seed, device)