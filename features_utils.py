from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import torch.nn.functional as F
import torch.optim as optim
import random

def make_femnist_datasets(X, y, train, K=10, seed=42, sigma=0.1):

    # 1) Group example‐indices by writer
    by_writer = defaultdict(list)
    for idx, writer in enumerate(train['writers']):
        by_writer[int(writer)].append(idx)

    # 2) shuffle writer IDs and split into K groups
    writer_ids = list(by_writer.keys())
    random.seed(seed)
    random.shuffle(writer_ids)
    per = len(writer_ids) // K
    groups = [writer_ids[i*per : (i+1)*per] for i in range(K-1)]
    groups.append(writer_ids[(K-1)*per :])

    # 3) for each group, collect X_i, y_i
    datalist = []
    for group_idx, group in enumerate(groups):
        idxs = [i for w in group for i in by_writer[w]]
        Xi = X[idxs]   # shape [n_i, ...]
        yi = y[idxs]   # shape [n_i,]
        noise_std = (group_idx / float(K)) * sigma
        noise = torch.randn_like(Xi) * noise_std
        Xi_noisy = Xi + noise
        Xi = torch.clamp(Xi_noisy, 0.0, 1.0)
        datalist.append((Xi, yi))

    return datalist

class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MADE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))



class WeightEstimator(nn.Module):
    """
    Estimates the weight α(x) = P(l=1 | u) / (1 - P(l=1 | u))
    based on MADE log-likelihood vectors.
    """
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(h)).squeeze(-1)
    
def train_local_made(model, loader, epochs=5, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, _ in loader:
            out = model(x)
            loss = F.binary_cross_entropy(out, x)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.state_dict()

def aggregate_models(states, weights):
    return {k: sum(weights[i] * states[i][k] for i in range(len(states)))
            for k in states[0]}

def train_global_made(loaders, dim, hid, rounds=10, local_epochs=1):
    gm = MADE(dim, hid)
    for _ in range(rounds):
        states, sizes = [], []
        for ld in loaders:
            lm = MADE(dim, hid)
            lm.load_state_dict(gm.state_dict())
            sd = train_local_made(lm, ld, epochs=local_epochs)
            states.append(sd); sizes.append(len(ld.dataset))
        total = sum(sizes)
        gm.load_state_dict(aggregate_models(states, [s/total for s in sizes]))
    return gm

def compute_sample_weights(global_made, local_made, loader,
                           device='cpu', num_epochs=1, lr=1e-3):
    """
    Trains the WeightEstimator to distinguish global vs local MADE log-likelihoods
    and computes sample weights α for all samples in loader.

    Args:
        global_made, local_made: MADE models; their forward(x) returns logits for Bernoulli outputs.
        loader: DataLoader yielding (X, _) batches (X in [0,1]).
        device: 'cpu' or 'cuda'.
        num_epochs: number of training epochs for estimator.
        lr: learning rate.

    Returns:
        Tensor of α weights for all samples in order.
    """
    # Determine input dimension
    sample_batch = next(iter(loader))[0].to(device)
    with torch.no_grad():
        logits = global_made(sample_batch)
    input_dim = logits.size(1)

    estimator = WeightEstimator(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)

    # Train
    estimator.train()
    for _ in range(num_epochs):
        for X, _ in loader:
            X = X.to(device)
            X_bin = (X >= 0.5).float()
            with torch.no_grad():
                ug = Bernoulli(logits=global_made(X)).log_prob(X_bin)
                ul = Bernoulli(logits=local_made(X)).log_prob(X_bin)
            U = torch.cat([ug, ul], dim=0)
            labels = torch.cat([
                torch.zeros(ug.size(0)),
                torch.ones(ul.size(0))
            ]).to(device)
            preds = estimator(U)
            loss = criterion(preds, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Compute α for each sample
    estimator.eval()
    alphas = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            X_bin = (X >= 0.5).float()
            ul = Bernoulli(logits=local_made(X)).log_prob(X_bin)
            p = estimator(ul)
            alphas.append((p / (1 - p)).cpu())
    return torch.cat(alphas)

