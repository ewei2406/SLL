import torch
from torch_geometric.nn import DenseGCNConv
import torch.nn.functional as F
from tqdm import tqdm


class GCN(torch.nn.Module):
    def __init__(self, input_features, output_classes, hidden_layers,  
        device, lr=0.01, dropout=0.5, weight_decay=5e-4, name=""):

        super(GCN, self).__init__()

        self.conv1 = DenseGCNConv(input_features, hidden_layers)
        self.conv2 = DenseGCNConv(hidden_layers, output_classes)

        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.name = name
        self.device = device
        self.to(device)

    def forward(self, x, adj):
        # x, edge_index = data.x, data.edge_index

        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)

        return F.log_softmax(x, dim=1).squeeze()
    
    def fitManual(self, features, adj, labels, idx_train, idx_test, epochs, verbose=True):

        if epochs == 0:
            return None

        # features = features
        # adj = adj
        # labels = labels

        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not verbose)
        t.set_description(f"Training {self.name}")

        for epoch in t:
            optimizer.zero_grad()
            predictions = self(features, adj)
            
            loss = F.cross_entropy(predictions[idx_train], labels[idx_train])
            
            loss.backward()
            optimizer.step()
            t.set_postfix({"loss": round(loss.item(), 2)})
        
        return predictions

    def fit(self, graph, epochs, verbose=True):

        if epochs == 0:
            return None

        features = graph.features
        adj = graph.adj
        labels = graph.labels

        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not verbose)
        t.set_description(f"Training {self.name}")

        for epoch in t:
            optimizer.zero_grad()
            predictions = self(features, adj)
            
            loss = F.cross_entropy(predictions[graph.idx_train], labels[graph.idx_train])
            
            loss.backward()
            optimizer.step()
            t.set_postfix({"loss": round(loss.item(), 2)})
        
        return predictions

    def train1epoch(self, features, adj, labels, idx_train, idx_test):

        features = features
        adj = adj
        labels = labels

        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        optimizer.zero_grad()
        predictions = self(features, adj)
        
        loss = F.cross_entropy(predictions[idx_train], labels[idx_train])
        
        loss.backward()
        optimizer.step()

        return loss.item()

    # def train1epoch(self, graph):
    #     features = graph.features
    #     adj = graph.adj
    #     labels = graph.labels

    #     self.train()
    #     optimizer = torch.optim.Adam(
    #         self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    #     optimizer.zero_grad()
    #     predictions = self(features, adj)
        
    #     loss = F.cross_entropy(predictions[graph.idx_train], labels[graph.idx_train])
        
    #     loss.backward()
    #     optimizer.step()

    #     return loss.item()


class GCN_continuous:
    def __init__(self, input_features, hidden_layers,  
        device, lr=0.01, dropout=0.5, weight_decay=5e-4, name=""):

        super(GCN, self).__init__()

        self.conv1 = DenseGCNConv(input_features, hidden_layers)
        self.conv2 = DenseGCNConv(hidden_layers, 1)

        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.name = name
        self.device = device
        self.to(device)

    def forward(self, x, adj):
        # x, edge_index = data.x, data.edge_index

        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)

        return x.squeeze()
    
    def fitManual(self, features, adj, labels, idx_train, idx_test, epochs, verbose=True):

        if epochs == 0:
            return None

        # features = features
        # adj = adj
        # labels = labels

        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not verbose)
        t.set_description(f"Training {self.name}")

        for epoch in t:
            optimizer.zero_grad()
            predictions = self(features, adj)
            
            loss = F.cross_entropy(predictions[idx_train], labels[idx_train])
            
            loss.backward()
            optimizer.step()
            t.set_postfix({"loss": round(loss.item(), 2)})
        
        return predictions
    
    def fit(self, graph, epochs, verbose=True):

        if epochs == 0:
            return None

        features = graph.features
        adj = graph.adj
        labels = graph.labels

        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not verbose)
        t.set_description(f"Training {self.name}")

        for epoch in t:
            optimizer.zero_grad()
            predictions = self(features, adj)
            
            loss = F.cross_entropy(predictions[graph.idx_train], labels[graph.idx_train])
            
            loss.backward()
            optimizer.step()
            t.set_postfix({"loss": round(loss.item(), 2)})
        
        return predictions

