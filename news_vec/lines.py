

import numpy as np

import torch
from torch import nn
from torch.nn.utils import rnn

from .cuda import itype, ftype


class LineEncoder(nn.Module):

    def __init__(self, input_size, hidden_size=1024, num_layers=2):
        """Initialize LSTM.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout()

    @property
    def out_dim(self):
        return self.lstm.hidden_size * 2

    def forward(self, x):
        """Sort, pack, encode, reorder.

        Args:
            x (list<Tensor>): Variable-length embedding tensors.
        """
        sizes = list(map(len, x))

        # Indexes to sort descending.
        sort_idxs = np.argsort(sizes)[::-1]

        # Indexes to restore original order.
        unsort_idxs = torch.from_numpy(np.argsort(sort_idxs)).type(itype)

        # Sort by size descending.
        x = [x[i] for i in sort_idxs]

        # Pad + pack, LSTM.
        x = rnn.pack_sequence(x)
        _, (hn, _) = self.lstm(x)

        # Cat forward + backward hidden layers.
        x = torch.cat([hn[0,:,:], hn[1,:,:]], dim=1)
        x = self.dropout(x)

        return x[unsort_idxs]
