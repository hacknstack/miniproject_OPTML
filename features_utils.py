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
from utils import *

def noise_adder(Xi,noise_std):
    noise = np.random.randn(*Xi.shape) * noise_std
    Xi_noisy = Xi + noise
    return np.clip(Xi_noisy, 0.0, 1.0)

def make_femnist_datasets(X, y, train, K=10, seed=42, sigma=0.5):

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
        Xi = noise_adder(Xi,noise_std)
        datalist.append((Xi, yi))

    return datalist

def evaluate_loss(model, X, y, alpha=None):
    """Mean cross‑entropy loss on (X,y)."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        if(alpha == None):
            return F.cross_entropy(
                logits,
                y,
                reduction='mean'
            ).item()
        per_sample = F.cross_entropy(logits, y, reduction='none')
        # weighted sum then normalize
        return (per_sample * alpha).sum().item() / alpha.sum().item()

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__() 
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_representation(self,x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return x

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias=bias)
        # Register mask as a buffer so it's moved with the model
        self.register_buffer('mask', mask)

    def forward(self, x):
        # Apply mask to weights before linear operation
        mask = self.mask.to(self.weight.device)
        return F.linear(x, self.weight * mask, self.bias)

class MADE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 1) Create a random ordering of the input indices 1..D
        # We'll assign each input feature a degree from 1..D
        self.register_buffer('input_degrees', torch.randperm(input_dim) + 1)

        # 2) Assign each hidden unit a random degree from 1..(D-1)
        # Hidden degrees must be between 1 and D-1
        hidden_degrees = np.random.randint(1, input_dim, size=hidden_dim)
        self.register_buffer('hidden_degrees', torch.from_numpy(hidden_degrees))

        # 3) Output degrees are fixed: for autoregressive order, we want output d to have degree = degree(input d)
        # Actually, output d should be degree(d) = input degree of that feature
        self.register_buffer('output_degrees', self.input_degrees)

        # 4) Create masks for W1 (hidden x input) and W2 (output x hidden)
        # mask1[k, j] = 1 if input_degree[j] > hidden_degree[k]
        mask1 = (self.input_degrees.unsqueeze(0) > self.hidden_degrees.unsqueeze(1)).float()
        # mask2[i, k] = 1 if hidden_degree[k] >= output_degree[i]
        mask2 = (self.hidden_degrees.unsqueeze(0) >= self.output_degrees.unsqueeze(1)).float()

        # Convert masks to same dtype as weights
        mask1 = mask1.to(torch.float32)
        mask2 = mask2.to(torch.float32)

        # 5) Define masked linear layers
        self.fc1 = MaskedLinear(input_dim, hidden_dim, mask1)
        self.fc2 = MaskedLinear(hidden_dim, input_dim, mask2)

    def forward(self, x):
        # x: [batch_size, input_dim]
        h = F.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(h))
        return out



class WeightEstimator(nn.Module):
    """
    Estimates the weight alpha(x) = P(l=1 | u) / (1 - P(l=1 | u))
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
    and computes sample weights alpha for all samples in loader.

    Args:
        global_made, local_made: MADE models; their forward(x) returns logits for Bernoulli outputs.
        loader: DataLoader yielding (X, _) batches (X in [0,1]).
        device: 'cpu' or 'cuda'.
        num_epochs: number of training epochs for estimator.
        lr: learning rate.

    Returns:
        Tensor of alpha weights for all samples in order.
    """
    # Determine input dimension
    sample_batch = next(iter(loader))[0].to(device)
    global_made = global_made.to(device)
    local_made  = local_made.to(device)
    
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
                ug = Bernoulli(probs=global_made(X)).log_prob(X_bin)
                ul = Bernoulli(probs=local_made(X)).log_prob(X_bin)
            U = torch.cat([ul, ug], dim=0)
            labels = torch.cat([
                torch.zeros(ul.size(0),device = device),
                torch.ones(ug.size(0),device = device)
            ]).to(device)
            preds = estimator(U)
            loss = criterion(preds, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Compute alpha for each sample
    estimator.eval()
    alphas = []
    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            X_bin = (X >= 0.5).float()
            ul = Bernoulli(probs=local_made(X)).log_prob(X_bin)
            p = estimator(ul)
            alphas.append((p / (1 - p)).cpu())
    return torch.cat(alphas)

def gd_step(model, data, target, alpha, gamma,device):
    # Compute weighted cross-entropy loss
    model.train()
    output = model(data).to(device)
    target = target.to(device)
    alpha = alpha.to(device)
    # reduction='none' to get per-sample losses
    loss_per_sample = F.cross_entropy(output, target, reduction='none')
    # Multiply by alpha and take mean
    weighted_loss = (alpha * loss_per_sample).mean()
    weighted_loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= gamma * param.grad
    return model


def client_update(model, data, target, alpha, K, gamma,device):
    # Run exactly K gradient steps using sample weights alpha
    for _ in range(K):
        model = gd_step(model, data, target, alpha, gamma,device)
    return model

def fedavg_disk(datalist, alphas_list, client_sizes, T, K, gamma,device = 'cpu',print_every=None):
    """
    Perform FedAvg with data-size weighting and sample-weighted loss.

    Args:
      datalist: list of tuples (X_tensor, y_tensor) per client
      alphas_list: list of alpha tensors (shape same as y_tensor) per client
      client_sizes: list of int N_k for each client (length K)
      T: number of communication rounds
      K: number of local GD steps per client per round
      gamma: learning rate for local updates

    Returns:
      global_model: trained global PyTorch model
    """
    n_clients = len(datalist)
    total_samples = sum(client_sizes)
    # Initialize global model
    global_model = SimpleNN()
    global_state = global_model.state_dict()
    #logging the loss
    loss_curve = []

    # Precompute weights N_k / N
    weights = [Nk / total_samples for Nk in client_sizes]

    for t in range(T):  #global rounds
        local_states = []
        client_losses = []
        # Broadcast & local training
        for i in range(n_clients):
            client_model = SimpleNN()
            client_model.load_state_dict(deepcopy(global_state))
            X_i, y_i = datalist[i]
            alpha_i = alphas_list[i]
            # Ensure data on same device as model
            X_i = torch.tensor(X_i, dtype=torch.float32).to(device)
            y_i = torch.tensor(y_i, dtype=torch.long).to(device)
                        # 3c) Clip & renormalize α to avoid extremely large weights
            if not isinstance(alpha_i, torch.Tensor):
                alpha_i = torch.tensor(alpha_i, dtype=torch.float32).to(device)
            alpha_i = torch.clamp(alpha_i, max=10.0)         # clip step
            alpha_i = alpha_i * (len(alpha_i) / alpha_i.sum())         # now sum(alpha_i)== N_k
            # Perform K local steps
            updated_model = client_update(client_model, X_i, y_i, alpha_i, K, gamma,device)
            local_states.append(deepcopy(updated_model.state_dict()))

                        # evaluate training loss on this client's data
            loss_i = evaluate_loss(client_model, X_i, y_i,alpha_i)
            client_losses.append(loss_i)
            local_states.append(deepcopy(client_model.state_dict()))

        # Aggregate weighted by client_sizes
        new_global_state = deepcopy(global_state)

        #log the average training loss this round
        # log the weighted training loss this round
        avg_loss = sum(weights[i] * client_losses[i] for i in range(n_clients))
        loss_curve.append(avg_loss)

        if print_every is not None :
            if (t + 1) % print_every == 0 or t == T - 1:
                print(f"  Avg training loss at round {t + 1}: {avg_loss:.4f}")

        for key in global_state.keys():
            # Weighted sum of parameters
            new_global_state[key] = sum(weights[i] * local_states[i][key] for i in range(n_clients))
        global_state = new_global_state
    
    global_model.load_state_dict(global_state)
    return global_model, loss_curve

def run_experiment_disk(datalist, T=50, K=5, gamma=0.01, weights=None, print_every=None, plot_loss=True, estimate_rate=True):
    model, loss_curve = fedavg_disk(datalist, T, K, gamma, print_every = 100, weights = None)
    
    if plot_loss:
        plt.figure(figsize=(8, 5))
        plt.plot(loss_curve, label="Training Loss")
        plt.xlabel("Round")
        plt.ylabel("Average Loss")
        plt.title("Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Log-log plot for convergence rate visualization
        plt.figure(figsize=(8, 5))
        rounds = np.arange(1, T + 1)
        loss_clipped = np.clip(loss_curve, 1e-10, None)
        plt.plot(np.log10(rounds), np.log10(loss_clipped), label="log-log Loss")
        plt.xlabel("log10(Round)")
        plt.ylabel("log10(Loss)")
        plt.title("Log-Log Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.show()

    if estimate_rate:
        rounds = np.arange(1, T + 1)
        rate = estimate_convergence_rate(rounds, loss_curve)
        print(f"Estimated convergence rate (slope on log-log scale): {rate:.4f}")

    return model, loss_curve