import torch

x = torch.Tensor(5,3)
print(x)
print(x.size())

y = torch.rand(5,3)
print(y)

print(y[1,:])

x = torch.ones(5)
x = x.numpy()
print(type(x))

print(torch.cuda.is_available())