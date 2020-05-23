"""Custom Pytorch Models
"""
import torch.nn as nn


class ReLU_net(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(ReLU_net, self).__init__()

        relus = []
        relus.append(nn.Flatten())
        relus.append(nn.Linear(input_size, hidden_sizes[0]))
        relus.append(nn.ReLU())

        for i in range(1, len(hidden_sizes)):
            relus.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            relus.append(nn.ReLU())

        relus.append(nn.Linear(hidden_sizes[-1], output_size))

        self.relus = nn.Sequential(*relus)

    def forward(self, x):
        x = self.relus(x)
        return x
