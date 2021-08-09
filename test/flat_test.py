import torch

test_tensor = torch.tensor([[[[1, 1, 1], [3, 3, 3]], [[4, 4, 4], [6, 6, 6]],[[13, 13, 13], [3, 3, 3]], [[4, 4, 4], [6, 6, 6]]]])
print(torch.flatten(test_tensor, start_dim=-2))
