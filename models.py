from torch import nn
import torch 


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, hidden3):
        super(AutoEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2, self.classifier(x1)



def calculate_loss(preds, actual, alpha=0.8):
    xpred, ypred = preds 
    xtrue, ytrue = actual

    reconstruct_loss_fn = torch.nn.MSELoss()
    classification_loss_fn = torch.nn.CrossEntropyLoss()

    l1 = reconstruct_loss_fn(xtrue, xpred)
    l2 = classification_loss_fn(ytrue, ypred)
    meta = dict(reconstruction_loss=l1, classification_loss=l2, total=l1+l2)
    return (1-alpha)*l1 + alpha*l2, meta




