# copied from: https://github.com/chenyuntc/pytorch-book
# chapter 2

# includes
from __future__ import print_function
from torch.autograd import Variable
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# define a 5*3 matrix
# allocate space without initialization
x = t.Tensor(5, 3)
x = t.rand(5, 3)

print(x.size())
print(x.size(1))

# support '+' operator
y = t.ones(5, 3)
print(x + y)

# operations on tensors are the same as those on numpy arrays
print(x[:, 1])

# convert tensors to numpy arrays
print(y.numpy())

# convert numpy arrays to tensors
a = np.ones(5)
b = t.from_numpy(a)
print(b)

# functions ended with '_' will change the parameter itself
b.add(1)
print(b)
b.add_(1)
print(b)

# tensors and numpy arrays share the same memory
# just like '&' in C++
print(a)

# auto differentiating
# Variables have auto differentiating, tensors not
x = Variable(t.ones(2, 2), requires_grad = True)
y = x.sum()
print(y)

print(y.grad_fn)

# BP for gradient
# gradient value is accumulated with the process of BP
# gradient value should be set zero before BP
for i in range(3):
    y.backward()
    print(x.grad)

# set zero
x.grad.data.zero_()
print(x.grad)
