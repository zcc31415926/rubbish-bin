# copied from: https://github.com/chenyuntc/pytorch-book
# chapter 2

# includes
from __future__ import print_function
from torch.autograd import Variable
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# neural networks:
# define a neural network class
class Net(nn.Module):
    def __init__(self):
        # equal to nn.Module.__init__(self)
        super(Net, self).__init__()

        # '1' represents the input graphs are single-channel
        # '6' is the number of output channels
        # '5' is the scale of the conv kernel (5*5)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # fully-connected layer (y = Wx + b)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # conv -> activate -> pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # size = -1: self-adapting to the input images
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# output the information of the network
net = Net()
print(net)

# output the values of the network parameters
for name, parameters in net.named_parameters():
    print(name, ':', parameters.size())

# '1' is the batch size
# '1' is the number of input channels
# (32, 32) is the (height, width) of the input images
input = Variable(t.randn(1, 1, 32, 32))

# preparation for BP
net.zero_grad()
out.backward(Variable(t.ones(1, 10)))

output = net(input)
target = Variable(t.arange(0, 10))
# MSELoss is the loss of Mean Square Error
# CrossEntropyLoss is the loss of Cross Entropy
criterion = nn.MSELoss()
loss = criterion(output, target)

# run the 'backward' step and compare the gradient before and afterwards
net.zero_grad()
print('before the backward process:')
print(net.conv1.bias.grad)
loss.backward()
print('after the backward process:')
print(net.conv1.bias.grad)

# define an optimizer and determine its target parameters, learning rate and momentum
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
optimizer.zero_grad()

output = net(input)
loss = criterion(output, target)
# run the 'backward' process
loss.backward()
# update the parameters of the optimizer
optimizer.step()

# 8 threads at one time
t.set_num_threads(8)
