import torch
import torch.nn as nn


class FinalLayer(nn.Module):
    def __init__(self, num_classes, in_filters, dropout):
        super(FinalLayer, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features=in_filters, out_features=num_classes)
        self.drop_out=nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        if self.dropout != 0:
            x = self.drop_out(x)
        return x

def finallayer(**kwargs):
    """
    Constructs the final layer.
    """
    return FinalLayer(**kwargs)