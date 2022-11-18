import torch

# a = torch.randn(3,4)
# print(a)

# change_a = a.transpose(0,1)
# print(change_a)

list_a = torch.tensor([[1,2,3,4], [5,6,7,8]])

print(list_a[..., :-1])