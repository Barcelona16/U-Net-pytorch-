import torch
import numpy as np

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

y = x

# print(y.grad_fn)

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
#print(y.data.norm()) #范数
while y.data.norm() < 1000:
    y = y * 2
y.backward(torch.ones_like(x))
# print(x.grad)

# print(x[...,2])

x = torch.rand(3,3)
y = torch.rand_like(x)
z = torch.rand_like(x)
#print(x,y,z)
t1 = torch.cat([x,y,z], dim = 1)
print(t1.shape)
print(t1)

y1 = t1 @ t1.T  #矩阵乘法
y2 = t1.matmul(t1.T)  
y3 = torch.rand_like(t1)
torch.matmul(t1, t1.T, out = y3)
print("- - - --  - - - - -- - - - -- -\n")
print(y1, y2, y3)

print(t1 * t1) # 点乘

agg = t1.sum()
print(agg) # tensor
print(agg.item()) # value

# tensor 2 numpy
n1 = t1.numpy()
print(n1.shape)

# numpy 2 tensor
n2 = np.ones(2)
t2 = torch.from_numpy(n2)
print(t2)
