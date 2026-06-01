"""
Example ECG model backbone placeholder.
Filename convention: model_<backbone>.py
"""

import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)
