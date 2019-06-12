import torch.nn.functional as F
from torch import nn
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, no_weight_norm):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, no_weight_norm=no_weight_norm)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)