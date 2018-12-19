

import torch
import numpy as np

from sklearn import metrics
from cached_property import cached_property
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from . import logger, utils
from .cuda import itype, ftype
from .model import Classifier


class ProgressDataLoader(DataLoader):

    def __iter__(self):
        """Track # generated pairs.
        """
        self.n = 0

        for x, y in super().__iter__():
            self.n += len(x)
            yield x, y


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
        self.es_wait = es_wait
        self.batch_size = batch_size

        token_counts = self.corpus.token_counts()
        labels = self.corpus.labels()

        # Initialize model.
        self.model = Classifier(labels, token_counts, **(model_kwargs or {}))

        if torch.cuda.is_available():
            self.model.cuda()

        # Initialize optimizer + loss.
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_func = nn.NLLLoss()

        # Set train / val / test splits.
        train_size = len(self.corpus) - (test_size * 2)
        sizes = (train_size, test_size, test_size)

        self.train_ds, self.val_ds, self.test_ds = \
            random_split(self.corpus, sizes)

        self.train_losses, self.val_losses = [], []

        # Pairs trained since last eval.
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

        loader = ProgressDataLoader(
            self.train_ds,
            collate_fn=self.model.collate_batch,
            batch_size=self.batch_size,
        )

        for lines, yt in loader:

            self.model.train()
            self.optimizer.zero_grad()

            yp = self.model(lines)

            loss = self.loss_func(yp, yt)
            loss.backward()

            self.optimizer.step()

            self.train_losses.append(loss.item())

            self.n += len(lines)

            if self.n >= self.eval_every:

                logger.info('Evaluating: %d / %d' %
                    (loader.n, len(self.train_ds)))

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
        for lines, yti in loader:
            yp += self.model(lines).tolist()
            yt += yti.tolist()

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
            logger.info('Val accuracy: %f' % preds.accuracy)

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
