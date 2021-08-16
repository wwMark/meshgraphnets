import torch

t = torch.randint(high=10, size=(3, 4))
print(t)
print(t[:, 0])