import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy




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

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.long)

        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_tensor).float().mean().item()
    return accuracy


