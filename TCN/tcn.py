import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, fixup_levels = None):
        super(TemporalBlock, self).__init__()
        self.fixup_levels = fixup_levels

        self.bias1 = nn.Parameter(torch.zeros(1))
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.subnet1 = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

        self.bias2 = nn.Parameter(torch.zeros(1))
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.subnet2 = nn.Sequential(self.conv2, self.chomp2, self.relu2, self.dropout2)


        self.downsample = weight_norm(nn.Conv1d(n_inputs, n_outputs, 1)) if n_inputs != n_outputs else None
        self.scale = nn.Parameter(torch.ones(1))
        self.bias3 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        if self.fixup_levels:
            self.conv1.weight.data.normal_(0, (2.0 / (self.conv1.kernel_size[0] * self.conv1.out_channels * self.fixup_levels)) ** 0.5)
            self.conv2.weight.data.zero_()
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, (2.0 / (self.downsample.kernel_size[0] * self.downsample.out_channels)) ** 0.5)
        else:#Fallback to normal init
            self.conv1.weight.data.normal_(0, 0.01)
            self.conv2.weight.data.normal_(0, 0.01)
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.subnet1(x + self.bias1)
        out = self.subnet2(out + self.bias2)
        out = out * self.scale + self.bias3

        res = x if self.downsample is None else self.downsample(x + self.bias1)
        
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, use_fixup_init=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        fixup_levels = num_levels if use_fixup_init else None
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout, fixup_levels=fixup_levels)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)