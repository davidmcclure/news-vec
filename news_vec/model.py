

import torch
import numpy as np
import string

from torch import nn
from torchtext.vocab import Vocab, Vectors
from torch.nn.utils import rnn
from torch.nn import functional as F
from itertools import chain

from . import utils
from . import settings
from .cuda import itype


class PretrainedTokenEmbedding(nn.Module):

    def __init__(self, token_counts, vector_file='glove.840B.300d.txt',
        vocab_size=10000, freeze=False):
        """Load pretrained embeddings.
        """
        super().__init__()

        self.vocab = Vocab(
            token_counts,
            vectors=Vectors(vector_file),
            max_size=vocab_size,
        )

        self.embed = nn.Embedding.from_pretrained(self.vocab.vectors, freeze)

        self.out_dim = self.embed.weight.shape[1]

    def forward(self, tokens):
        """Map to token embeddings.
        """
        x = [self.vocab.stoi[t] for t in tokens]
        x = torch.LongTensor(x).type(itype)

        return self.embed(x)


# CharCNN params from https://arxiv.org/abs/1508.06615

class CharEmbedding(nn.Embedding):

    def __init__(self, embed_dim=15):
        """Set vocab, map s->i.
        """
        self.vocab = (
            string.ascii_letters +
            string.digits +
            string.punctuation
        )

        # <PAD> -> 0, <UNK> -> 1
        self._ctoi = {s: i+2 for i, s in enumerate(self.vocab)}

        super().__init__(len(self.vocab)+2, embed_dim)

    def ctoi(self, c):
        return self._ctoi.get(c, 1)

    def chars_to_idxs(self, chars, max_size=20):
        """Map characters to embedding indexes.
        """
        # Truncate super long tokens, to prevent CUDA OOMs.
        chars = chars[:max_size]

        idxs = [self.ctoi(c) for c in chars]

        return torch.LongTensor(idxs).type(itype)

    def forward(self, tokens, min_size=7):
        """Batch-embed token chars.

        Args:
            tokens (list<str>)
        """
        # Map chars -> indexes.
        xs = [self.chars_to_idxs(t) for t in tokens]

        pad_size = max(min_size, max(map(len, xs)))

        # Pad + stack index tensors.
        x = torch.stack([
            F.pad(x, (0, pad_size-len(x)))
            for x in xs
        ])

        return super().forward(x)


class CharCNN(nn.Module):

    def __init__(self, widths=range(1, 7), fpn=25, out_dim=512):
        """Conv layers + linear projection.
        """
        super().__init__()

        self.embed = CharEmbedding()

        self.widths = widths

        self.convs = nn.ModuleList([
            nn.Conv2d(1, w*fpn, (w, self.embed.weight.shape[1]))
            for w in self.widths
        ])

        conv_dims = sum([c.out_channels for c in self.convs])

        self.out = nn.Linear(conv_dims, out_dim)

        self.out_dim = out_dim

    def forward(self, tokens):
        """Convolve, max pool, linear projection.
        """
        x = self.embed(tokens, max(self.widths))

        # 1x input channel.
        x = x.unsqueeze(1)

        # Convolve, max pool.
        x = [torch.tanh(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]

        # Cat filter maps.
        x = torch.cat(x, 1)

        return self.out(x)


class TokenEmbedding(nn.Module):

    def __init__(self, token_counts):
        """Initialize token + char embeddings
        """
        super().__init__()

        self.embed_t = PretrainedTokenEmbedding(token_counts)
        self.embed_c = CharCNN()
        self.dropout = nn.Dropout()

        self.out_dim = self.embed_t.out_dim + self.embed_c.out_dim

    def forward(self, tokens):
        """Map to token embeddings, cat with character convolutions.
        """
        # Token embeddings.
        xt = self.embed_t(tokens)

        # Char embeddings.
        xc = self.embed_c(tokens)
        x = torch.cat([xt, xc], dim=1)

        x = self.dropout(x)

        return x


class TokenCNN(nn.Module):

    # https://www.aclweb.org/anthology/D14-1181

    def __init__(self, input_size, filter_size=settings.CNN_FILTER_SIZE,
        filter_widths=settings.CNN_FILTER_WIDTHS):
        """Initialize convolutions.
        """
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(1, filter_size, (w, input_size))
            for w in filter_widths
        ])

        self.out_dim = sum([c.out_channels for c in self.convs])

    def forward(self, x):
        """Convolve, max pool, linear projection.
        """
        # 1x input channel.
        x = x.unsqueeze(1)

        # Convolve, max pool.
        x = [torch.tanh(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]

        # Cat filter maps.
        x = torch.cat(x, 1)

        return x


class TokenLSTM(nn.Module):

    def __init__(self, input_size, hidden_size=settings.LSTM_HIDDEN_SIZE,
        num_layers=settings.LSTM_NUM_LAYERS):
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

        self.out_dim = self.lstm.hidden_size * 2

    def forward(self, x):
        """Sort, pack, encode, reorder.

        Args:
            x (list<Tensor>): Variable-length embedding tensors.

        Returns:
            x (Tensor)
            states (list<Tensor): LSTM states per input.
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
        states, (hn, _) = self.lstm(x)

        # Unpack + unpad states.
        states, _ = rnn.pad_packed_sequence(states, batch_first=True)
        states = [t[:size] for t, size in zip(states[unsort_idxs], sizes)]

        # Cat forward + backward hidden layers.
        x = torch.cat([hn[0,:,:], hn[1,:,:]], dim=1)
        x = x[unsort_idxs]

        return x, states


class LineEncoderCBOW(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.out_dim = input_size

    def forward(self, x):
        x = torch.stack([xi.mean(0) for xi in x])
        return x


class LineEncoderLSTM(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.lstm = TokenLSTM(input_size)
        self.out_dim = self.lstm.out_dim

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class LineEncoderCNN(TokenCNN):

    def forward(self, x):
        x = rnn.pad_sequence(x, batch_first=True)
        return super().forward(x)


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size=settings.ATTN_HIDDEN_SIZE):

        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.out_dim = input_size

    def forward(self, states):
        """Score states, regroup by seq, linearly combine states.
        """
        sizes = list(map(len, states))

        attn = self.score(torch.cat(states))

        attn = [
            F.softmax(scores.squeeze(), dim=0)
            for scores in utils.group_by_sizes(attn, sizes)
        ]

        states_attn = torch.stack([
            torch.sum(si * ai.view(-1, 1), 0)
            for si, ai in zip(states, attn)
        ])

        return states_attn


class LineEncoderLSTMAttn(nn.Module):

    def __init__(self, input_size):
        """Initialize LSTM + attention.
        """
        super().__init__()
        self.lstm = TokenLSTM(input_size)
        self.attn = Attention(self.lstm.out_dim)
        self.out_dim = self.lstm.out_dim + self.attn.out_dim

    def forward(self, x):
        """Sort, pack, encode, reorder.

        Args:
            x (list<Tensor>): Variable-length embedding tensors.
        """
        x, states = self.lstm(x)
        states_attn = self.attn(states)

        # Tops + state attn.
        x = torch.cat([x, states_attn], dim=1)

        return x


class Classifier(nn.Module):

    @classmethod
    def from_dataset(cls, ds, *args, **kwargs):
        """Build from HeadlineDataset.
        """
        token_counts = ds.token_counts()
        labels = ds.labels()
        return cls(labels, token_counts, *args, **kwargs)

    def __init__(self, labels, token_counts, line_enc,
        embed_dim=settings.CLF_EMBED_DIM):
        """Initialize encoders + clf.
        """
        super().__init__()

        self.labels = labels

        self.ltoi = {label: i for i, label in enumerate(labels)}

        self.embed_tokens = TokenEmbedding(token_counts)

        # TODO: Better way to handle this?

        if line_enc == 'cbow':
            self.encode_lines = LineEncoderCBOW(self.embed_tokens.out_dim)

        elif line_enc == 'lstm':
            self.encode_lines = LineEncoderLSTM(self.embed_tokens.out_dim)

        elif line_enc == 'cnn':
            self.encode_lines = LineEncoderCNN(self.embed_tokens.out_dim)

        elif line_enc == 'lstm-attn':
            self.encode_lines = LineEncoderLSTMAttn(self.embed_tokens.out_dim)

        self.merge = nn.Linear(self.encode_lines.out_dim, embed_dim)

        self.dropout = nn.Dropout()

        self.predict = nn.Sequential(
            nn.Linear(embed_dim, len(labels)),
            nn.LogSoftmax(1),
        )

    def embed(self, lines):
        """Embed lines.
        """
        tokens = [line['clf_tokens'] for line in lines]

        # Line lengths.
        sizes = [len(ts) for ts in tokens]

        # Embed tokens, regroup by line.
        x = self.embed_tokens(list(chain(*tokens)))
        x = utils.group_by_sizes(x, sizes)

        # Embed lines.
        x = self.encode_lines(x)

        # Blend encoder outputs, dropout.
        x = self.merge(x)
        x = self.dropout(x)

        return x

    def forward(self, lines):
        return self.predict(self.embed(lines))

    def collate_batch(self, batch):
        """Labels -> indexes.
        """
        lines, labels = list(zip(*batch))

        yt_idx = [self.ltoi[label] for label in labels]
        yt = torch.LongTensor(yt_idx).type(itype)

        return lines, yt
