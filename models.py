
from turtle import forward
from matplotlib import collections
from torch import nn, sigmoid
import torch 
from torch.utils.data import DataLoader, Dataset

import metrics

from tqdm import tqdm 
class DatasetWithDataLoader(Dataset):
    def get_dataloader(self, batch_size=16, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class XOnly(DatasetWithDataLoader):
    def __init__(self, x):
        self.x = torch.Tensor(x)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index]

# class NenvDataset(Dataset):
#     def __init__(self, graph_node_features, labels, mask=None, genes=None, graph=None):
#         self.x = torch.Tensor(graph_node_features)
#         self.y = torch.Tensor(labels)
#         self.genes = genes
#         self.graph = graph
#         self.mask = mask if mask is None else torch.Tensor(mask)

#     def __len__(self):
#         return len(self.x)
    
#     def __getitem__(self, index):
#         if self.mask is not None:
#             return self.x[index, :], self.y[index, :], self.mask[index]
#         return self.x[index, :], self.y[index, :]


### getting models for training

# class AutoEncoderOnly(nn.Module):
#     def __init__(self, input_dim, hidden_layers, hidden2):
#         super(AutoEncoderOnly, self).__init__()
#         self.flatten = nn.Flatten()
#         self.input_dim = input_dim

#         hidden_layers = [input_dim, *hidden_layers, hidden2]
#         layers = []
#         for i, o in zip(hidden_layers, hidden_layers[1:]):
#             layers.append(nn.Linear(i, o))
#             layers.append(nn.Sigmoid())
        
#         self.encoder = nn.Sequential(
#             *layers
#         )

#         hidden_layers = hidden_layers[::-1]
#         layers = []
#         for i, o in zip(hidden_layers, hidden_layers[1:]):
#             layers.append(nn.Linear(i, o))
#             layers.append(nn.Sigmoid())

#         layers[-1] = nn.Sigmoid()

#         self.decoder = nn.Sequential(
#             *layers
#         )
    
#     def forward(self, x):
#         x1 = self.encoder(x)
#         x2 = self.decoder(x1)
#         return x2
    
#     def encode(self, x):
#         return self.encoder(x)


class AutoEncoderOnly(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden2):
        super(AutoEncoderOnly, self).__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden2, input_dim)
        )

        
    
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2
    
    def encode(self, x):
        return self.encoder(x)


class Classifier(nn.Module):
    def __init__(self, encoder_model, hidden2, output_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder_model

        self.encoder_layer_out = encoder_model[-1].out_features

        self.dnn = nn.Sequential(
            nn.Linear(self.encoder_layer_out, hidden2),
            nn.Relu(),
            nn.Linear(hidden2, output_classes),
            nn.sigmoid()
        )

    
    def forward(self, x):
        enc_x = self.encode(x)
        return self.dnn(enc_x)
    

def to_device(x):
    return x.to('cuda' if torch.cuda.is_available() else 'cpu')
class Trainer:
    def __init__(self, model):
        self.model = to_device(model)
        self.metrics = []
        self.loss_each_step = dict(train={}, valid={})

    def compile(self, lossfn, optimizer, lr=0.01):
        self.loss_fn = lossfn 
        self._optfn = optimizer
        self.default_lr = lr

        self.opt = self._optfn(self.model.parameters(), lr=self.default_lr)

    def _update_lr(self, lr):
        for p in self.opt.param_groups:
            p['lr'] = lr

    

    def train(self, train_data, valid_data, epochs=500, batch_size=128, lr=None, logevery=100, lr_scheduler=None, gn=0.1*0.5):
        train_dataloader = train_data.get_dataloader(batch_size=batch_size)
        valid_dataloader = valid_data.get_dataloader(batch_size=batch_size)

        lr = lr or self.default_lr
        if self.default_lr != lr: self._update_lr(lr)

        # self.loss_each_step = dict(train={}, valid={})
        mean = lambda x: sum(x)/len(x)
        # self.metrics = []

        if lr_scheduler is not None:
            lr_schedular = lr_scheduler(self.opt)

        for e in tqdm(range(epochs)):
            self.model.train()
            self.loss_each_step['train'][e] = []
            rankmetric = metrics.RankMetric()
            recallat5 = metrics.RecallAtN(n=5)
            # avgrank = metrics.AvgRank()
            for ds in train_dataloader:
                x = to_device(ds)
                x += gn * torch.randn(x.size())
                
                self.opt.zero_grad()
                x_dash = self.model(x)
                loss = self.loss_fn(x, x_dash)
                loss.backward()
                self.opt.step()

                self.loss_each_step['train'][e].append(loss.item())

            if e%logevery== 0:
                print(f'epoch train loss: {mean(self.loss_each_step["train"][e])}')

            self.model.eval()
            self.loss_each_step['valid'][e] = []

            for ds in valid_dataloader:
                x = to_device(ds)
                
                with torch.no_grad():
                    x_dash = self.model(x)
                    loss = self.loss_fn(x, x_dash)
                
                xnp, xdashnp = x.numpy(), x_dash.numpy()
                rankmetric(xnp, xdashnp)
                recallat5(xnp, xdashnp)

                
                self.loss_each_step['valid'][e].append(loss.item())

            if e%logevery == 0:
                print(f'epoch valid loss: {mean(self.loss_each_step["valid"][e])}')
                print(f'epoch Rank metric: {rankmetric.aggr():.4f}')
                print(f'epoch Avg Rank metric: {mean(rankmetric.collect):.4f}')
                print(f'epoch Recall-at-5 metric: {recallat5.aggr():.4f}')
                self.metrics.extend([rankmetric, recallat5])
                print(f'current lr: {self.opt.param_groups[0]["lr"]}')
            
                if lr_schedular:
                    lr_schedular.step()
            

class TrainerClassifer:
    def __init__(self, model):
        self.model = to_device(model)
        self.metrics = []
        self.loss_each_step = dict(train={}, valid={})

    def compile(self, lossfn, optimizer, lr=0.01):
        self.loss_fn = lossfn 
        self._optfn = optimizer
        self.default_lr = lr

        self.opt = self._optfn(self.model.parameters(), lr=self.default_lr)

    def _update_lr(self, lr):
        for p in self.opt.param_groups:
            p['lr'] = lr

    

    def train(self, train_data, valid_data, epochs=500, batch_size=128, lr=None, logevery=100, lr_scheduler=None, gn=0.1*0.5):
        train_dataloader = train_data.get_dataloader(batch_size=batch_size)
        valid_dataloader = valid_data.get_dataloader(batch_size=batch_size)

        lr = lr or self.default_lr
        if self.default_lr != lr: self._update_lr(lr)

        # self.loss_each_step = dict(train={}, valid={})
        mean = lambda x: sum(x)/len(x)
        # self.metrics = []

        if lr_scheduler is not None:
            lr_schedular = lr_scheduler(self.opt)

        for e in tqdm(range(epochs)):
            self.model.train()
            self.loss_each_step['train'][e] = []
            rankmetric = metrics.RankMetric()
            recallat5 = metrics.RecallAtN(n=5)
            # avgrank = metrics.AvgRank()
            for dsx, dsy in train_dataloader:
                x = to_device(dsx)
                y = to_device(dsy)
                
                x += gn * torch.randn(x.size())
                
                self.opt.zero_grad()
                x_dash = self.model(x)
                loss = self.loss_fn(x, x_dash)
                loss.backward()
                self.opt.step()

                self.loss_each_step['train'][e].append(loss.item())

            if e%logevery== 0:
                print(f'epoch train loss: {mean(self.loss_each_step["train"][e])}')

            self.model.eval()
            self.loss_each_step['valid'][e] = []

            for ds in valid_dataloader:
                x = to_device(ds)
                
                with torch.no_grad():
                    x_dash = self.model(x)
                    loss = self.loss_fn(x, x_dash)
                
                xnp, xdashnp = x.numpy(), x_dash.numpy()
                rankmetric(xnp, xdashnp)
                recallat5(xnp, xdashnp)

                
                self.loss_each_step['valid'][e].append(loss.item())

            if e%logevery == 0:
                print(f'epoch valid loss: {mean(self.loss_each_step["valid"][e])}')
                print(f'epoch Rank metric: {rankmetric.aggr():.4f}')
                print(f'epoch Avg Rank metric: {mean(rankmetric.collect):.4f}')
                print(f'epoch Recall-at-5 metric: {recallat5.aggr():.4f}')
                self.metrics.extend([rankmetric, recallat5])
                print(f'current lr: {self.opt.param_groups[0]["lr"]}')
            
                if lr_schedular:
                    lr_schedular.step()
            




def to_device(x):
    return x.to('cuda' if torch.cuda.is_available() else 'cpu')


# def train_model(model, train_dataloader, valid_dataloader, epochs, lr):
#     opt = torch.optim.Adam(model.parameters(), lr=0.1)
#     metric_stores = []


#     for e in range(epochs):
#         model.train()
#         for ds in train_dataloader:
#             x, y = map(to_device, ds)
#             opt.zero_grad()
#             xp, yp = model.forward(x)
#             loss = models.calculate_loss((xp, yp), (x, y))
#             loss.backward()
#             opt.step()

#             # log loss with meta 
        

#         model.eval()
        
#         m = MetricStore()
#         with torch.no_grad():
#             total_loss = 0
            
#             for ds in valid_dataloader:
#                 x, y = map(to_device, ds)
#                 xp, yp = model.forward(x)
#                 val_loss = models.calculate_loss((xp, yp), (x, y))
#                 m(y.numpy(), yp.numpy().round())
#                 total_loss += val_loss
#             print(f'metric_store_pred: {m.aggr()}')
#             metric_stores.append(m)

# class AutoEncoder(nn.Module):
#     def __init__(self, input_dim, hidden1, hidden2, hidden3, numclasses):
#         super(AutoEncoder, self).__init__()
#         self.flatten = nn.Flatten()
#         self.input_dim = input_dim
#         self.numclasses = numclasses


#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden1),
#             nn.ReLU(),
#             nn.Linear(hidden1, hidden2),
#             nn.ReLU()
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(hidden2, hidden1),
#             nn.ReLU(),
#             nn.Linear(hidden1, input_dim),
#             nn.ReLU(),
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(hidden2, hidden3),
#             nn.ReLU(),
#             nn.Linear(hidden3, numclasses),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x1 = self.encoder(x)
#         x2 = self.decoder(x1)
#         return x2, self.classifier(x1)



# def calculate_loss(preds, actual, alpha=0.8):
#     xpred, ypred = preds 
#     xtrue, ytrue = actual

#     reconstruct_loss_fn = torch.nn.MSELoss()
#     classification_loss_fn = torch.nn.BCELoss()

#     l1 = reconstruct_loss_fn(xpred, xtrue)
#     l2 = classification_loss_fn(ypred, ytrue)
#     # meta = dict(reconstruction_loss=l1, classification_loss=l2, total=l1+l2)
#     return (1-alpha)*l1 #+ alpha*l2, meta


# def calculate_masked_loss(preds, actual, mask, alpha=0.8):
#     xpred, ypred = preds 
#     xtrue, ytrue = actual

#     reconstruct_loss_fn = torch.nn.MSELoss()
#     classification_loss_fn = torch.nn.CrossEntropyLoss()

#     ypred1 = ypred[mask]
#     ytrue1 = ytrue[mask]

#     l1 = reconstruct_loss_fn(xtrue, xpred)
#     l2 = classification_loss_fn(ytrue1, ypred1)

#     meta = dict(reconstruction_loss=l1, classification_loss=l2, total=l1+l2)
#     return (1-alpha)*l1 + alpha*l2, meta
