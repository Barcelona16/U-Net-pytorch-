import torch

x = torch.Tensor(5,3)
# print(x)
# print(x.size())

# y = torch.rand(5,3)
# print(y)

# print(y[1,:])

# x = torch.ones(5)
# x = x.numpy()
# print(type(x))

# print(torch.cuda.is_available())

from torch.autograd import Variable
x = Variable(torch.ones(2,2), requires_grad = True)
print(x)

y = x
print(y)

print(y.grad_fn)

z = y * y * 3
#out = z.mean()
# print(y.grad_fn) # 叶子节点为None 结果节点为梯度函数类型
# print(y.is_leaf) # 是否为叶子节点
#print(type(z))
z.backward(torch.ones_like(x))  #因为z不是标量 backward需要参数
#print(type(z))
#z.backward()


x = torch.randn(3)
x = Variable(x, requires_grad = True)

y = x * 2
print(y)
print(y.data.norm())
while y.data.norm() < 1000:
    y = y * 2

print(y)