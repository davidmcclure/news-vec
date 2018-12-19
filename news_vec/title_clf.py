

import ujson
import gzip
import string
import math
import pickle
import random
import os
# import sys
# import re

import numpy as np

from itertools import islice, chain
from collections import Counter, defaultdict
from glob import glob
from tqdm import tqdm
from boltons.iterutils import pairwise, chunked_iter
from sklearn.model_selection import train_test_split
from sklearn import metrics
from cached_property import cached_property

import torch
from torchtext.vocab import Vocab, Vectors
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import rnn

from . import logger, utils
from .cuda import itype, ftype
from .tokens import TokenEmbedding
from .lines import LineEncoder


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
        x = utils.group_by_sizes(x, sizes)

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


class Predictions:

    def __init__(self, yt, yp):
        self.yt = yt
        self.yp = yp

    def __repr__(self):
        return f'{self.__class__.__name__}<{len(self.yt)}>'

    @cached_property
    def accuracy(self):
        return metrics.accuracy_score(self.yt, self.yp.argmax(1))


class EarlyStoppingException(Exception):
    pass


class Trainer:

    def __init__(self, corpus, eval_every, test_size, es_wait, lr=1e-4,
        batch_size=50, model_kwargs=None):

        self.corpus = corpus
        self.eval_every = eval_every
        self.test_size = test_size
        self.es_wait = es_wait
        self.batch_size = batch_size
        self.lr = lr

        token_counts = self.corpus.token_counts()
        labels = self.corpus.labels()

        self.model = Classifier(labels, token_counts, **(model_kwargs or {}))
        self.loss_func = nn.NLLLoss()

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, max_epochs=100):
        """Train for N epochs.
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        train_size = len(self.corpus) - (self.test_size * 2)
        sizes = (train_size, self.test_size, self.test_size)

        self.train_ds, self.val_ds, self.test_ds = \
            random_split(self.corpus, sizes)

        self.train_losses, self.val_losses = [], []
        self.n = 0

        for epoch in range(max_epochs):

            # Listen for early stopping exception.
            try:
                self.train_epoch(epoch)
            except EarlyStoppingException:
                logger.info('Stopping early.')
                break

    def train_epoch(self, epoch):

        logger.info('Epoch %d' % epoch)

        loader = DataLoader(
            self.train_ds,
            collate_fn=self.model.collate_batch,
            batch_size=self.batch_size,
        )

        eval_n = 0
        for i, (lines, yt) in enumerate(loader):

            self.model.train()
            self.optimizer.zero_grad()

            yp = self.model(lines)

            loss = self.loss_func(yp, yt)
            loss.backward()

            self.optimizer.step()

            self.train_losses.append(loss.item())

            self.n += len(lines)

            utils.print_replace(loader.batch_size * (i+1))

            if self.n >= self.eval_every:

                print('\r')

                logger.info(
                    'Evaluating: %d / %d' %
                    (loader.batch_size * (i+1), len(self.train_ds))
                )

                self.validate()
                self.n = 0

                if self.is_finished():
                    raise EarlyStoppingException()

    def _predict(self, split):
        """Predict a val/test split.
        """
        self.model.eval()

        loader = DataLoader(
            split,
            collate_fn=self.model.collate_batch,
            batch_size=self.batch_size,
        )

        yt, yp = [], []
        for i, (lines, yti) in enumerate(loader):
            yp += self.model(lines).tolist()
            yt += yti.tolist()
            utils.print_replace(loader.batch_size * (i+1))

        print('\r')

        yt = torch.LongTensor(yt).type(itype).cpu()
        yp = torch.FloatTensor(yp).type(ftype).cpu()

        return Predictions(yt, yp)

    def validate(self, log=True):

        preds = self._predict(self.val_ds)

        loss = self.loss_func(preds.yp, preds.yt)
        self.val_losses.append(loss.item())

        if log:
            recent_tl = np.mean(self.train_losses[-100:])
            logger.info('Train loss: ~%f' % recent_tl)
            logger.info('Val loss: %f' % self.val_losses[-1])
            logger.info('Val acc: %f' % preds.accuracy)

    def eval_test(self):
        return self._predict(self.test_ds)

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

            embeds = utils.tensor_to_np(embeds)
            yps = utils.tensor_to_np(yps)

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
