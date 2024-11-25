import torch
import numpy as np
from torch import nn


class cross_entropy_loss():
    def __init__(self):
        super(cross_entropy_loss, self).__init__()
        self.eps = torch.tensor(np.finfo(float).eps)

    def loss(self, ypred, ytruth):
        cross_entropy = -torch.mean(ytruth * torch.log(ypred + self.eps))
        return cross_entropy


class FCLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        # self.dropout=nn.Dropout(p=0.3, inplace=False)
        self.bnorm = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU(inplace=True)
        # self.residual = input_size == output_size

    def forward(self, x):
        x = self.fc(x)
        x = self.bnorm(x)
        x = self.relu(x)
        return x


def mixup(x, shuffle, lam, i, j):
    if shuffle is not None and lam is not None and i == j:
        x = lam * x + (1 - lam) * x[shuffle, :]
    return x


class Mixup_Model(nn.Module):
    def __init__(self, num_classes, inputsize, mixup_layers=None, bottleneck_size=2):
        print(bottleneck_size)
        super(Mixup_Model, self).__init__()
        self.sizes = [inputsize, 128, 64, 32, 16, bottleneck_size, 16, 32, 64, 128]
        self.out_layers = [0, 4, 8]
        self.numlayers = len(self.sizes)
        if mixup_layers is None:
            mixup_layers = []
            for i in range(len(self.sizes)):
                # print(i)
                # if(self.sizes[i]>self.sizes[0]):
                if self.sizes[i] > num_classes and i != len(self.sizes) // 2:
                    mixup_layers.append(i)
        self.mixup_layers=mixup_layers
        layers = []
        for i in range(self.numlayers-1):
            layers.append(FCLayer(self.sizes[i], self.sizes[i+1]))
        self.layers = nn.ModuleList(layers)
        self.projection = nn.Linear(self.sizes[-1], num_classes)

    def forward(self, x, start_layer=0):
        if isinstance(x, list):
            x, shuffle, lam, mixup_layer = x
        else:
            shuffle = None
            lam = None
            mixup_layer = None
        # Decide which layer to mixup
        layer_out={}
        for k in range(start_layer, self.numlayers):
            x = mixup(x, shuffle, lam, k, mixup_layer)
            if(k<self.numlayers-1):
                x = self.layers[k](x)
            else:
                x = self.projection(x)
            if(k in self.out_layers):
                layer_out[k]=x

        return (x, layer_out)


def mixup_model(num_classes, inputsize, mixup_layers=None, bottleneck_size=2):
    model = Mixup_Model(num_classes, inputsize, mixup_layers=mixup_layers, bottleneck_size=bottleneck_size)
    return model