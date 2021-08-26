import torch

a = torch.tensor([1], device='cuda')
print("a device", a.device)
b = torch.tensor(a, device='cpu')
print("b device", b.device)
