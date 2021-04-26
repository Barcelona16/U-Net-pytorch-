'''
Author: Deavan
Date: 2021-04-26 08:30:19
Description: 
'''
import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

#print(w, b)

# print('Gradient function for z =',z.grad_fn)
# print('Gradient function for loss =', loss.grad_fn)

loss.backward()
print("loss is  ", loss, "\n")
print(w.grad)
print(b.grad)

# with torch.no_grad():
#     z = torch.matmul(x, w) + b
# print(z.requires_grad)



