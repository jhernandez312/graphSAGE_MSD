import torch
from torch.nn import Linear
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, layer_type, in_channels, out_channels, hidden_channels=16, n_hidden=2):
        super(Model, self).__init__()
        torch.manual_seed(42)

        self.is_mlp = layer_type.__name__ == 'Linear'

        self.layer1 = layer_type(in_channels, hidden_channels)
        self.layer2 = torch.nn.ModuleList()

        for _ in range(n_hidden - 1):
            self.layer2.append(layer_type(hidden_channels, hidden_channels))

        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.layer1(x) if self.is_mlp else self.layer1(x, edge_index)
        h = F.relu(h)

        for layer in self.layer2:
            h = layer(h) if self.is_mlp else layer(h, edge_index)
            h = F.relu(h)

        out = self.classifier(h)
        return out