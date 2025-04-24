import torch
from torch import nn, optim
from torchvision import datasets, transforms 
import torch.nn.functional as F

torch.set_default_tensor_type(torch.cuda.FloatTensor)  # To tackle an error with pysyft not acceptin
