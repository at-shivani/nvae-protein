#### read graphs as networkx 

import networkx as nx
from sklearn.metrics import precision_recall_fscore_support
import torch
import readutils
from collections import defaultdict

NUM_PROTEINS = 18362

def read_newtork_from_edgelist_txt(path, asint=True):
    textlist = readutils.readtextfile(path)
    graph = nx.Graph()
    graph.add_nodes_from(list(range(NUM_PROTEINS)))
    name = readutils.getfilename(path)
    for i in range(len(textlist)):
        s, t, *w = readutils.split_line(textlist[i], dlim='\t')
        if asint:
            _ = graph.add_edge(int(s)-1, int(t)-1, weight=float(w[0]) if w else 1.)
        else:
            _ = graph.add_edge(s, t, weight=float(w[0]) if w else 1.)
    graph.name = name 
    return graph

def read_labels(label_path, gene_key_path, function_key_path, only_keep_func_path):
    adj = readutils.readtextfile(label_path)
    adj = [readutils.split_line(s1, dlim='\t') for s1 in adj]
    adj  = [[int(i), int(j)] for i, j in adj]

    func_keys = readutils.readtextfile(function_key_path)
    gene_keys = readutils.readtextfile(gene_key_path)
    only_keep_func = readutils.readtextfile(only_keep_func_path)
    func_mapping = [[gene_keys[i-1], func_keys[j-1]] for i, j in adj]

    m = defaultdict(set)
    for g, f in func_mapping:
        if f in only_keep_func:
            m[g].add(f)
    
    for g in m:
        m[g] = list(m[g])
    
    return m


def prepare_dataset(graph, label_dict, genes_path, as_one_hot=False):
    genes = readutils.readtextfile(genes_path)
    m = {n: (label_dict.get(genes[n], None) or []) for n in range(NUM_PROTEINS) if n in graph.nodes()}

    # print(len(m))

    if as_one_hot:
        func_list = list({v for vs in label_dict.values() for v in vs})
        labels = np.zeros((len(m), len(func_list)))
        for i, fs in enumerate(m.values()):
            for f in fs:
                labels[i, func_list.index(f)] = 1.
    
        return labels
    return m 



### learn structural features of the graph

from node2vec import Node2Vec
import numpy as np

class GraphFeatures:
    DIMENSIONS = 100
    
    class Training:
        WALK_LENGTH = 15
        NUM_WALKS = 200
        WORKERS = 4
        WINDOW = 5 
        BATCH = 4

    


def learn_node_features(graph):
    node2vec = Node2Vec(
        graph, dimensions=GraphFeatures.DIMENSIONS, walk_length=GraphFeatures.Training.WALK_LENGTH,
        num_walks=GraphFeatures.Training.NUM_WALKS, workers=GraphFeatures.Training.WORKERS)  # Use temp_folder for big graphs
    model = node2vec.fit(window=GraphFeatures.Training.WINDOW, min_count=1, batch_words=GraphFeatures.Training.BATCH)
    # print({type(node) for node in graph.nodes()})
    # return model
    return np.stack([model.wv[str(node)] for node in graph.nodes()])


### Dataset

from torch.utils.data import Dataset

class NenvDataset(Dataset):
    def __init__(self, graph_node_features, labels, mask=None, genes=None, graph=None):
        self.x = torch.Tensor(graph_node_features)
        self.y = torch.Tensor(labels)
        self.genes = genes
        self.graph = graph
        self.mask = mask if mask is None else torch.Tensor(mask)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        if self.mask is not None:
            return self.x[index, :], self.y[index, :], self.mask[index]
        return self.x[index, :], self.y[index, :]



### getting models for training

import models

HIDDEN_LAYERS1 = 25
HIDDEN_LAYERS2 = 10
HIDDEN_LAYERS3 = 50  
NUM_CLASSES = 50 # number of label classes

def get_autoencoder_model():
    return models.AutoEncoder(GraphFeatures.DIMENSIONS, HIDDEN_LAYERS1, HIDDEN_LAYERS2, HIDDEN_LAYERS3, NUM_CLASSES)

def to_device(x):
    return x.to('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_dataloader, valid_dataloader, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    metric_stores = []


    for e in range(epochs):
        model.train()
        for ds in train_dataloader:
            x, y = map(to_device, ds)
            opt.zero_grad()
            xp, yp = model.forward(x)
            loss = models.calculate_loss((xp, yp), (x, y))
            loss.backward()
            opt.step()

            # log loss with meta 
        

        model.eval()
        
        m = MetricStore()
        with torch.no_grad():
            total_loss = 0
            
            for ds in valid_dataloader:
                x, y = map(to_device, ds)
                xp, yp = model.forward(x)
                val_loss = models.calculate_loss((xp, yp), (x, y))
                m(y.numpy(), yp.numpy().round())
                total_loss += val_loss
            print(f'metric_store_pred: {m.aggr()}')
            metric_stores.append(m)

class MetricStore:
    def __init__(self, mode='macro'):
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.global_count = 0
        self.mode = mode
    
    def _update(self, precision, recall, f1, n=None):
        self.precision += precision
        self.recall += recall 
        self.f1 += f1 
        if self.mode == 'micro': assert isinstance(n, int)
        else: n = 1
        self.global_count += 1 if n is None else n
    
    def __call__(self, ytrue, ypred):
        p, r, f1, _ = precision_recall_fscore_support(ytrue, ypred, average=self.mode)
        n = len(ytrue)
        self._update(p, r, f1, n=n)


    def aggr(self):
        assert self.global_count > 0
        return dict(
            precision=self.precision/self.global_count,
            recall=self.recall/self.global_count,
            f1=self.f1/self.global_count)

    def __str__(self):
        return str(self.aggr())



# still to figure
# - test model and training and code written so far end to end with dummy data
# - metric 
# - finish essential remaining part e.g. saving and reloading 
# - inferencing 
# - train valid test split 
# full training and improvements
