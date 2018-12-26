

import torch
import numpy as np

from sklearn import metrics
from cached_property import cached_property
from torch import nn, optim
from torch.utils.data import random_split

from . import logger
from .cuda import itype, ftype
from .utils import ProgressDataLoader
from .model import Classifier


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

    def __init__(self, model, corpus, es_wait=5, eval_every=None,
        lr=1e-4, batch_size=50):

        self.model = model

        if torch.cuda.is_available():
            self.model.cuda()

        self.corpus = corpus
        self.batch_size = batch_size
        self.es_wait = es_wait

        # Initialize optimizer + loss.
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_func = nn.NLLLoss()

        # By default, eval after each epoch.
        self.eval_every = eval_every or len(self.corpus.train)

        # Store loss histories.
        self.train_losses, self.val_losses = [], []

        # Pairs trained since last eval.
        self.n_from_eval = 0

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
            self.corpus.train,
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

            self.n_from_eval += len(lines)

            if self.n_from_eval >= self.eval_every:

                logger.info(f'Evaluating: {epoch} | {loader.n}')
                self.validate()

                if self.is_finished():
                    raise EarlyStoppingException()

    def _predict(self, split):
        """Predict a val/test split.
        """
        self.model.eval()

        loader = ProgressDataLoader(
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

        preds = self._predict(self.corpus.val)

        loss = self.loss_func(preds.yp, preds.yt)
        self.val_losses.append(loss.item())

        if log:
            recent_tl = np.mean(self.train_losses[-10:])
            logger.info('Train loss: ~%f' % recent_tl)
            logger.info('Val loss: %f' % self.val_losses[-1])
            logger.info('Val accuracy: %f' % preds.accuracy)

        self.n_from_eval = 0

    def eval_test(self):
        return self._predict(self.corpus.test)

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
            # Earliest loss < window average.
            window[0] < np.mean(window)
        )
