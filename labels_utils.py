import numpy as np
import torch
import torch.nn.functional as F

def create_dirichlet_clients(X, y, n, beta):
    num_classes = 10
    class_indices = [np.where(y == i)[0] for i in range(num_classes)]

    client_data_indices = [[] for _ in range(n)]
    class_counts = [len(indices) for indices in class_indices]

    # For each class, distribute samples to clients using Dirichlet distribution
    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        
        proportions = np.random.dirichlet(np.repeat(beta, n))
        # Scale proportions so that each client gets len(X)//n samples eventually
        proportions = np.array([p * (len(indices) / (len(X) // n)) for p in proportions])
        proportions = (proportions / proportions.sum()).clip(0, 1)
        splits = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, splits)

        for i, client_idxs in enumerate(split_indices):
            client_data_indices[i].extend(client_idxs)

    # Truncate or pad to ensure each client has exactly len(X)//n samples
    max_len = len(X) // n
    for i in range(n):
        client_data_indices[i] = client_data_indices[i][:max_len]

    # Form the datalist
    datalist = [(X[indices], y[indices]) for indices in client_data_indices]
    return datalist

def compute_inverse_kl_weights(datalist, num_classes=10, epsilon=1e-8):
    # Compute global label distribution
    all_labels = np.concatenate([y for _, y in datalist])
    global_counts = np.bincount(all_labels, minlength=num_classes)
    global_dist = global_counts / global_counts.sum()

    kl_divs = []
    for _, y in datalist:
        client_counts = np.bincount(y, minlength=num_classes)
        client_dist = client_counts / client_counts.sum()

        # Add epsilon to avoid 0s and ensure numerical stability
        client_dist = np.clip(client_dist, epsilon, 1)
        global_dist_safe = np.clip(global_dist, epsilon, 1)
        
        kl = np.sum(global_dist_safe * np.log(global_dist_safe / client_dist))
        kl_divs.append(kl)
    
    # Inverse KL, add epsilon to avoid division by zero
    inv_kl = 1 / (np.array(kl_divs) + epsilon)

    # Normalize to sum to 1
    weights = inv_kl / inv_kl.sum()
    return weights


def moon_contrastive_loss(h_local, h_global, h_prev, temperature=0.5):
    """
    MOON contrastive loss. Moon will use the representation from the model i.e simple_NN

    h_local: current model's representation
    h_global: global model's representation
    h_prev: previous local model representation 
    """
    #normalization 
    h_local = F.normalize(h_local, dim=1)
    h_global = F.normalize(h_global, dim=1)
    h_prev = F.normalize(h_prev, dim=1)

    #compute similarities
    pos_sim = torch.sum(h_local * h_global, dim=1) / temperature
    neg_sim = torch.sum(h_local * h_prev, dim=1) / temperature

    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(h_local.size(0), dtype=torch.long, device=h_local.device)  # positives are class 0

    contrastive_loss = F.cross_entropy(logits, labels)
    return contrastive_loss

