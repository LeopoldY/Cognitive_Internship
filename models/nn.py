# Model
import torch.nn as nn

class sexnet(nn.Module):
    def __init__(self):
        super(sexnet, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(2, 2),
        )

    def forward(self, x):
        out = self.dense(x)
        return out