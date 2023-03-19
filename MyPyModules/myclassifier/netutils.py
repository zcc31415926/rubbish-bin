import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.backbone = net
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)

    def loss(self, x, y):
        pred = self.forward(x)
        return self.criterion(pred, y)

