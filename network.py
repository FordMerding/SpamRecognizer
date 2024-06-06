import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(180, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.LogSoftmax(dim = -1)
        )
    def forward(self, a):
        out = self.layers(a)
        return out