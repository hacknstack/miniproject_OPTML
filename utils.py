import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from labels_utils import moon_contrastive_loss



class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
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

    
def gd_step(model, data, target, gamma):
    model.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= gamma * param.grad
    return model

def client_update(model, data, target, K, gamma):
    for _ in range(K):
        model = gd_step(model, data, target, gamma)
    return model



#client update adaptation for MOON - Combines gd_step + client_update

def client_update_moon(model, global_model, prev_model, data, target, K, gamma, mu=1.0, temperature=0.5):
    model.train()
    for _ in range(K):
        model.zero_grad()
        output = model(data)
        loss_sup = F.cross_entropy(output, target)   #classic loss 

        # Representations
        h_local = model.get_representation(data)
        h_global = global_model.get_representation(data)
        h_prev = prev_model.get_representation(data)

        contrastive_loss = moon_contrastive_loss(h_local, h_global, h_prev, temperature)

        total_loss = loss_sup + mu * contrastive_loss  #Total loss, as explained in the MOON Paper
        total_loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= gamma * param.grad

    return model, total_loss.item()   # to return the model and the total loss. Because now the loss is not only the cross entropy loss


def fedavg(datalist, T, K, gamma, weights = None):
    n = len(datalist)
    n = len(datalist)
    if weights is None:
        weights = [1.0 / n] * n  # Equal contribution by default
    assert len(weights) == n, "Length of weights must match number of clients"
    assert abs(sum(weights) - 1.0) < 1e-5, "Weights must sum to 1"

    global_model = SimpleNN()
    global_model_state = deepcopy(global_model.state_dict())

    for t in range(T):
        print("round : ", t+1)
        local_states = []

        for i in range(n):
            client_model = SimpleNN()
            client_model.load_state_dict(global_model_state)

            X, y = datalist[i]
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)

            client_model = client_update(client_model, X_tensor, y_tensor, K, gamma)
            local_states.append(deepcopy(client_model.state_dict()))
        
        # Weighted Federated Averaging
        new_global_state = deepcopy(global_model_state)
        for key in global_model_state:
            new_global_state[key] = sum(weights[i] * local_states[i][key] for i in range(n))
        
        global_model_state = new_global_state

    global_model.load_state_dict(global_model_state)
    return global_model


# modif of fedavg with moon loss - MOON - Model contrastive federated learning

def fedavg_moon(datalist, T, K, gamma, mu=1.0, temperature=0.5, print_every=None, weights=None):
    n = len(datalist)
    if weights is None:
        weights = [1.0 / n] * n
    assert len(weights) == n, "Length of weights must match number of clients"
    assert abs(sum(weights) - 1.0) < 1e-5, "Weights must sum to 1"

    #global state
    global_model = SimpleNN()
    global_model_state = deepcopy(global_model.state_dict())
    prev_local_models = [deepcopy(global_model) for _ in range(n)]      #previous local models for each client to use in the contrastive loss

    #logging the loss
    loss_curve = []

    for t in range(T):  #global rounds
        #print(f"Round {t+1}/{T}")
        local_states = []
        client_losses = []

        #local rounds
        for i in range(n):
            client_model = SimpleNN()
            client_model.load_state_dict(global_model_state)

            X, y = datalist[i]
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)

            # updating the client model with the moon update : parameters = (model, global_model, prev_model, data, target, K, gamma, mu=1.0, temperature=0.5):
            client_updated_model, loss_i = client_update_moon(client_model, global_model, prev_local_models[i], X_tensor, y_tensor, K, gamma, mu=1.0, temperature=0.5)

            # evaluate training loss on this client's data
            client_losses.append(loss_i)
            local_states.append(deepcopy(client_model.state_dict()))
            prev_local_models[i] = deepcopy(client_updated_model)  # update for the next round, each previous local weights are used in the next iteration

        new_global_state = deepcopy(global_model_state)
        for key in global_model_state:
            new_global_state[key] = sum(weights [i] * local_states [i][key] for i in range(n))

        global_model_state = new_global_state

        
        # log the weighted training loss this round
        avg_loss = sum(weights[i] * client_losses[i] for i in range(n))
        loss_curve.append(avg_loss)


        if print_every is not None :
            if (t + 1) % print_every == 0 or t == T - 1:
                print(f"  Avg training loss at round {t + 1}: {avg_loss:.4f}")

    global_model.load_state_dict(global_model_state)

    return global_model, loss_curve

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.long)

        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_tensor).float().mean().item()
    return accuracy


#Similar but with loss log, to obtain convergence rates

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



def fedavg_loss(datalist, T, K, gamma, print_every=None, weights=None) :
    n = len(datalist)
    if weights is None:
        weights = [1.0 / n] * n
    assert len(weights) == n, "Length of weights must match number of clients"
    assert abs(sum(weights) - 1.0) < 1e-5, "Weights must sum to 1"

    #global state
    global_model = SimpleNN()
    global_model_state = deepcopy(global_model.state_dict())
    #logging the loss
    loss_curve = []

    for t in range(T):  #global rounds
        #print(f"Round {t+1}/{T}")
        local_states = []
        client_losses = []

        #local rounds
        for i in range(n):
            client_model = SimpleNN()
            client_model.load_state_dict(global_model_state)

            X, y = datalist[i]
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            client_model = client_update(client_model, X_tensor, y_tensor, K, gamma)

            # evaluate training loss on this client's data
            loss_i = evaluate_loss(client_model, X_tensor, y_tensor)
            client_losses.append(loss_i)
            local_states.append(deepcopy(client_model.state_dict()))

        new_global_state = deepcopy(global_model_state)
        for key in global_model_state:
            new_global_state[key] = sum(weights [i] * local_states [i][key] for i in range(n))

        global_model_state = new_global_state

        #log the average training loss this round
        # log the weighted training loss this round
        avg_loss = sum(weights[i] * client_losses[i] for i in range(n))
        loss_curve.append(avg_loss)


        if print_every is not None :
            if (t + 1) % print_every == 0 or t == T - 1:
                print(f"  Avg training loss at round {t + 1}: {avg_loss:.4f}")

    global_model.load_state_dict(global_model_state)

    return global_model, loss_curve


def compute_convergence(loss_curve):
    """
    Given L = [L0, L1, ..., L_{T-1}], returns:
      abs_dec[i] = L[i] - L[i+1]
      rel_dec[i] = (L[i] - L[i+1]) / L[i]
    for i = 0..T-2.
    """
    abs_dec = []
    for i in range(len(loss_curve) - 1):
        d = loss_curve[i] - loss_curve[i+1]
        abs_dec.append(d)
    return abs_dec


import matplotlib.pyplot as plt
import numpy as np
from utils import fedavg_loss  # Ensure this is properly defined and imported

def estimate_convergence_rate(rounds, loss_curve, fit_start_frac=0.5):
    """
    Estimate convergence rate from the slope of the log-log loss curve.
    
    Args:
        rounds: array of iteration indices (e.g., 1 to T)
        loss_curve: list of losses per round
        fit_start_frac: fraction of the curve from which to start fitting

    Returns:
        slope (float): estimated convergence rate
    """
    
    loss_gap = np.array([l for l in loss_curve])
    loss_gap = np.clip(loss_gap, a_min=1e-10, a_max=None)  # avoid log(0)

    start = int(len(rounds) * fit_start_frac)
    x = np.log10(rounds[start:])
    y = np.log10(loss_gap[start:])

    slope, _ = np.polyfit(x, y, deg=1)
    return slope


#Implement estimation of rate of convergence

def run_experiment(datalist, T=50, K=5, gamma=0.01, weights=None, print_every=None, plot_loss=True, estimate_rate=True):
    model, loss_curve = fedavg_loss(datalist, T, K, gamma, print_every = 100, weights = None)
    
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

#collecting final_losses for different values of T)

def sample_final_losses_vs_T(datalist, T_list, K, gamma, 
                        average_last_n=1,  #average over how many final rounds
                        weights=None):
    """
    Runs FedAvg for each T in T_list, using fixed K and gamma,
    then extracts a single “final loss” per run:
      - If average_last_n == 1 return loss_curve[-1].
      - Otherwise return the mean of loss_curve[-average_last_n :].
    
    Returns a dict { T: final_loss }.
    """
    results = {}
    for T in T_list:
        model, loss_curve = fedavg_loss(datalist, T=T, K=K, gamma=gamma, weights=weights)
        
        if average_last_n <= 1:
            final_loss = loss_curve[-1]
        else:
            final_loss = np.mean(loss_curve[-average_last_n : ])
        
        results[T] = final_loss
        print(f"  → T={T}, final averaged loss = {final_loss:.4f}")
    return results


def sample_final_losses_vs_K(datalist, T, K_list, gamma, 
                        average_last_n=1,  #average over how many final rounds
                        weights=None):
    
    results = {}
    for K in K_list:
        model, loss_curve = fedavg_loss(datalist, T=T, K=K, gamma=gamma, weights=weights)
        
        if average_last_n <= 1:
            final_loss = loss_curve[-1]
        else:
            final_loss = np.mean(loss_curve[-average_last_n : ])
        
        results[K] = final_loss
        print(f"  → K={K}, final averaged loss = {final_loss:.4f}")
    return results

