import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, no_weight_norm, use_fixup_init, num_levels, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.use_fixup_init = use_fixup_init
        self.num_levels = num_levels

        weight_norm = torch.nn.utils.weight_norm
        if no_weight_norm:
            weight_norm = lambda x: x

        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation, bias=False))
        self.chomp1 = Chomp1d(padding)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding,
                                           dilation=dilation, bias=False))
        self.chomp2 = Chomp1d(padding)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)


        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=False) if n_inputs != n_outputs else None
        self.scale = nn.Parameter(torch.ones(1))
        self.bias3b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        if self.use_fixup_init:
            self.conv1.weight.data.normal_(0, (2.0 / (self.conv1.kernel_size[0] * self.conv1.out_channels * self.num_levels)) ** 0.5)
            self.conv2.weight.data.zero_()
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, (2.0 / (self.downsample.kernel_size[0] * self.downsample.out_channels)) ** 0.5)
        else:
            self.conv1.weight.data.normal_(0, 0.01)
            self.conv2.weight.data.normal_(0, 0.01)
            if self.downsample is not None:
                self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.chomp1(self.conv1(x + self.bias1a))
        out = self.dropout1(self.relu1(out + self.bias1b))

        out = self.chomp2(self.conv2(out + self.bias2a))
        out = self.dropout2(self.relu2(out + self.bias2b))

        out = out * self.scale + self.bias3b

        res = x if self.downsample is None else self.downsample(x + self.bias1a)
        
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, no_weight_norm, use_fixup_init, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout,
                                     no_weight_norm=no_weight_norm, use_fixup_init=use_fixup_init, num_levels=num_levels)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)