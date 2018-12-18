

import ujson
import gzip
import string
import math
import pickle
import os
import re

import numpy as np

from collections import Counter
from itertools import islice, chain
from glob import glob
from tqdm import tqdm
from boltons.iterutils import pairwise, chunked_iter
from sklearn.model_selection import train_test_split
from sklearn import metrics

import torch
from torchtext.vocab import Vocab, Vectors
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import rnn

from . import logger
from .utils import group_by_sizes, tensor_to_np
from .cuda import itype, ftype


def read_json_lines(root):
    """Read JSON corpus.

    Yields: Line
    """
    for path in glob('%s/*.gz' % root):
        with gzip.open(path) as fh:
            for line in fh:

                data = ujson.loads(line)

                tokens = data.pop('tokens')

                if not tokens:
                    continue

                label = data.pop('label')

                yield Line(tokens, label, data)


class Line:

    def __init__(self, tokens, label=None, metadata=None):
        self.tokens = tokens
        self.label = label
        self.metadata = metadata or {}

    def __repr__(self):

        pattern = '{cls_name}<{token_count} tokens -> {label}>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            token_count=len(self.tokens),
            label=self.label,
        )

    def to_dict(self):
        return dict(tokens=self.tokens, label=self.label, **self.metadata)


class Corpus(Dataset):

    def __init__(self, root, skim=None):
        """Read lines.
        """
        logger.info('Parsing line corpus.')

        lines_iter = islice(read_json_lines(root), skim)

        self.lines = list(tqdm(lines_iter))

    def __repr__(self):

        pattern = '{cls_name}<{line_count} lines>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            line_count=len(self),
        )

    def __len__(self):
        return len(self.lines)

    def __iter__(self):
        return iter(self.lines)

    def __getitem__(self, i):
        """Get (tokens -> label idx) training pair.
        """
        line = self.lines[i]
        return (line, line.label)

    def token_counts(self):
        """Collect all token -> count.
        """
        logger.info('Gathering token counts.')

        counts = Counter()
        for line in tqdm(self):
            counts.update(line.tokens)

        return counts

    def label_counts(self):
        """Label -> count.
        """
        logger.info('Gathering label counts.')

        counts = Counter()
        for line in tqdm(self):
            counts[line.label] += 1

        return counts

    def labels(self):
        counts = self.label_counts()
        return [label for label, _ in counts.most_common()]


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

    @property
    def out_dim(self):
        return self.embed.weight.shape[1]

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

        # TODO: Zero UNK + PAD.

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

        self.dropout = nn.Dropout()

    @property
    def out_dim(self):
        return self.out.out_features

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
        x = self.dropout(x)

        return self.out(x)


class TokenEmbedding(nn.Module):

    def __init__(self, token_counts):
        """Initialize token + char embeddings
        """
        super().__init__()

        self.embed_t = PretrainedTokenEmbedding(token_counts)
        self.embed_c = CharCNN()

    @property
    def out_dim(self):
        return self.embed_t.out_dim + self.embed_c.out_dim

    def forward(self, tokens):
        """Map to token embeddings, cat with character convolutions.
        """
        # Token embeddings.
        xt = self.embed_t(tokens)

        # Char embeddings.
        xc = self.embed_c(tokens)
        x = torch.cat([xt, xc], dim=1)

        return x


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


class Classifier(nn.Module):

    def __init__(self, labels, token_counts, embed_dim=512, lstm_kwargs=None):
        """Initialize encoders + clf.
        """
        super().__init__()

        self.labels = labels

        self.ltoi = {label: i for i, label in enumerate(labels)}

        self.embed_tokens = TokenEmbedding(token_counts)

        self.embed_lines = LineEncoder(self.embed_tokens.out_dim,
            **(lstm_kwargs or {}))

        self.merge = nn.Linear(self.embed_lines.out_dim, embed_dim)

        self.predict = nn.Sequential(
            nn.Linear(embed_dim, len(labels)),
            nn.LogSoftmax(1),
        )

    def embed(self, lines):
        """Embed lines.
        """
        tokens = [line.tokens for line in lines]

        # Line lengths.
        sizes = [len(ts) for ts in tokens]

        # Embed tokens, regroup by line.
        x = self.embed_tokens(list(chain(*tokens)))
        x = group_by_sizes(x, sizes)

        # Embed lines.
        x = self.embed_lines(x)

        return self.merge(x)

    def forward(self, lines):
        return self.predict(self.embed(lines))

    def collate_batch(self, batch):
        """Labels -> indexes.
        """
        lines, labels = list(zip(*batch))

        yt_idx = [self.ltoi[label] for label in labels]
        yt = torch.LongTensor(yt_idx).type(itype)

        return lines, yt


class BarDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bar = tqdm(self.dataset)

    def __iter__(self):
        """Update progress bar.
        """
        for x, y in super().__iter__():
            yield x, y
            self.bar.update(len(x))

    @property
    def n(self):
        return self.bar.n


class EarlyStoppingException(Exception):
    pass


class Trainer:

    @classmethod
    def from_spark_json(cls, root, skim=None, *args, **kwargs):
        corpus = Corpus(root, skim)
        return cls(corpus, *args, **kwargs)

    def __init__(self, corpus, lr=1e-4, batch_size=50, test_size=10000,
        eval_every=100000, es_wait=5, corpus_kwargs=None, model_kwargs=None):

        self.corpus = corpus
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.es_wait = es_wait

        token_counts = self.corpus.token_counts()
        labels = self.corpus.labels()

        self.model = Classifier(labels, token_counts, **(model_kwargs or {}))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if torch.cuda.is_available():
            self.model.cuda()

        # Train / test split.
        s1, s2 = len(self.corpus) - test_size, test_size
        self.train_ds, self.val_ds = random_split(self.corpus, (s1, s2))

        self.train_losses, self.val_losses = [], []
        self.n = 0

    def train(self, max_epochs=100):
        """Train for N epochs.
        """
        for epoch in range(max_epochs):

            # Listen for early stopping exception.
            try:
                self.train_epoch(epoch)
            except EarlyStoppingException:
                logger.info('Stopping early.')
                break

    def train_epoch(self, epoch):

        logger.info('Epoch %d' % epoch)

        loader = BarDataLoader(
            self.train_ds,
            collate_fn=self.model.collate_batch,
            batch_size=self.batch_size,
        )

        eval_n = 0
        for lines, yt in loader:

            self.model.train()
            self.optimizer.zero_grad()

            yp = self.model(lines)

            loss = F.nll_loss(yp, yt)
            loss.backward()

            self.optimizer.step()

            self.train_losses.append(loss.item())

            self.n += len(lines)

            if self.n >= self.eval_every:

                self.validate()
                self.n = 0

                if self.is_finished():
                    raise EarlyStoppingException()

    def predict_val(self):
        """Predict val lines.
        """
        self.model.eval()

        loader = BarDataLoader(
            self.val_ds,
            collate_fn=self.model.collate_batch,
            batch_size=self.batch_size,
        )

        yt, yp = [], []
        for lines, yti in loader:
            yp += self.model(lines).tolist()
            yt += yti.tolist()

        yt = torch.LongTensor(yt).type(itype)
        yp = torch.FloatTensor(yp).type(ftype)

        return yt, yp

    def validate(self, log=True):

        yt, yp = self.predict_val()

        loss = F.nll_loss(yp, yt)
        self.val_losses.append(loss.item())

        if log:
            self.log_perf(yt, yp)

    def log_perf(self, yt, yp, ntl=100):

        # LOSS
        logger.info('Train loss: %f' % np.mean(self.train_losses[-ntl:]))
        logger.info('Val loss: %f' % self.val_losses[-1])

        # ACCURACY
        preds = yp.argmax(1)
        acc = metrics.accuracy_score(yt, preds)
        logger.info('Val acc: %f' % acc)

        yt_lb = [self.model.labels[i] for i in yt.tolist()]
        yp_lb = [self.model.labels[i] for i in preds.tolist()]

        # REPORT
        report = metrics.classification_report(yt_lb, yp_lb)
        logger.info('\n' + report)

    def is_finished(self):
        """Has val loss stalled?
        """
        window = self.val_losses[-self.es_wait:]
        window_mean = np.mean(window)

        window_strs = ' '.join(['{0:.5g}'.format(val) for val in window])
        window_mean_str = '{0:.5g}'.format(window_mean)

        logger.info(f'ES window: {window_strs} | {window_mean_str}')

        return (
            # Eval'ed at least N times.
            len(window) >= self.es_wait and
            # Window average is greater than earliest loss.
            window[0] < np.mean(window)
        )


def write_fs(path, data):
    """Dump data to disk.
    """
    logger.info('Flushing to disk: %s' % path)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as fh:
        fh.write(data)


class CorpusEncoder:

    def __init__(self, corpus, model, segment_size=1000, batch_size=100):
        """Wrap corpus + model.
        """
        self.corpus = corpus

        self.model = model
        self.model.eval()

        self.segment_size = segment_size
        self.batch_size = batch_size

    def preds_iter(self):
        """Generate encoded lines + metadata.
        """
        loader = BarDataLoader(
            self.corpus,
            collate_fn=self.model.collate_batch,
            batch_size=self.batch_size,
        )

        for lines, yt in loader:

            embeds = self.model.embed(lines)
            yps = self.model.predict(embeds).exp()

            embeds = tensor_to_np(embeds)
            yps = tensor_to_np(yps)

            for line, embed, yp in zip(lines, embeds, yps):

                preds = {
                    f'p_{domain}': mass
                    for domain, mass in zip(self.model.labels, yp)
                }

                # Metadata + clf output.
                data = dict(
                    **line.to_dict(),
                    **preds,
                    embedding=embed,
                )

                yield data

    def segments_iter(self):
        """Generate (fname, data).
        """
        chunks = chunked_iter(self.preds_iter(), self.segment_size)

        for i, lines in enumerate(chunks):
            fname = '%s.p' % str(i).zfill(5)
            yield fname, pickle.dumps(lines)

    def write_fs(self, root):
        """Flush to local filesystem.
        """
        for fname, data in self.segments_iter():
            path = os.path.join(root, fname)
            write_fs(path, data)
