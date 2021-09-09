import torch

a = torch.Tensor([[1], [2], [3]])
b = torch.Tensor(1)
print(torch.eq(a, b))