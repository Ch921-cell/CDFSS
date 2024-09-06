import torch
import numpy as np

arr = np.array([85,6,84,725,15,36,48,92,324])

t = torch.Tensor(arr)
t= t.view(3,3)

print(t)