

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
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import rnn

from . import logger
from .cuda import itype, ftype


SEP_TOKENS = {':', '-', '–', '—', '|', 'via', '[', ']'}


# TODO: Reuters hack for "brief-XXX". Learn with char LSTM?
def split_first_token(tokens):
    """If the first token has a hyphen, break into separate tokens.
    """
    first = re.split('(-)', tokens[0])
    return [*first, *tokens[1:]]


def scrub_paratext(tokens):
    """Try to prune out "paratext" around headlines. Hacky.
    """
    sep_idxs = [
        i for i, t in enumerate(tokens)
        if t.lower() in SEP_TOKENS
    ]

    if not sep_idxs:
        return tokens

    if sep_idxs[0] != 0:
        sep_idxs = [-1] + sep_idxs

    if sep_idxs[-1] != len(tokens)-1:
        sep_idxs = sep_idxs + [len(tokens)]

    widths = [
        (i1, i2, i2-i1)
        for i1, i2 in pairwise(sep_idxs)
    ]

    widths = sorted(
        widths,
        key=lambda x: x[2],
        reverse=True,
    )

    i1 = widths[0][0]+1
    i2 = widths[0][1]

    return tokens[i1:i2]


CURLY_STRAIGHT = (('“', '"'), ('”', '"'), ('‘', "'"), ('’', "'"))


def uncurl_quotes(text):
    """Curly -> straight.
    """
    for c, s in CURLY_STRAIGHT:
        text = text.replace(c, s)

    return text


QUOTES = {'\'', '"'}


def scrub_quotes(tokens):
    """Remove quote tokens.
    """
    return [t for t in tokens if uncurl_quotes(t) not in QUOTES]


def drop_empty_strings(tokens):
    """Remove empty tokens.
    """
    return [t for t in tokens if len(t)]


def clean_headline(tokens):
    """Raw tokens -> clf tokens.
    """
    tokens = split_first_token(tokens)
    tokens = scrub_paratext(tokens)
    tokens = scrub_quotes(tokens)
    tokens = drop_empty_strings(tokens)
    return tokens


class Line:

    def __init__(self, tokens, label, count, lower=True):
        self.tokens = [t.lower() for t in tokens] if lower else tokens
        self.label = label
        self.count = count

    def __repr__(self):

        pattern = '{cls_name}<{token_count} tokens -> {label}>'

        return pattern.format(
            cls_name=self.__class__.__name__,
            token_count=len(self.tokens),
            label=self.label,
        )


def read_json_lines(root, lower=True):
    """Read JSON corpus.

    Yields: Line
    """
    for path in glob('%s/*.gz' % root):
        with gzip.open(path) as fh:
            for line in fh:

                data = ujson.loads(line)
                print(data)

                tokens = data.get('tokens')
                tokens = clean_headline(tokens)

                if not tokens:
                    continue

                yield Line(
                    tokens,
                    data['label'],
                    data['count'],
                    lower=lower,
                )


class Corpus:

    def __init__(self, root, skim=None, lower=True):
        """Read lines.
        """
        logger.info('Parsing line corpus.')

        lines_iter = islice(read_json_lines(root, lower), skim)

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

    def __init__(self, embed_dim, lstm_dim, num_layers=1):
        """Initialize LSTM.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            embed_dim,
            lstm_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout()

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


def group_by_sizes(L, sizes):
    """Given a flat list and a list of sizes that sum to the length of the
    list, group the list into sublists with corresponding sizes.

    Args:
        L (list)
        sizes (list<int>)

    Returns: list<list>
    """
    parts = []

    total = 0
    for s in sizes:
        parts.append(L[total:total+s])
        total += s

    return parts


class Classifier(nn.Module):

    def __init__(self, labels, token_counts, lstm_dim, embed_dim):
        """Initialize encoders + clf.
        """
        super().__init__()

        self.labels = labels

        self.ltoi = {label: i for i, label in enumerate(labels)}

        self.embed_tokens = TokenEmbedding(token_counts)

        self.embed_lines = LineEncoder(self.embed_tokens.out_dim, lstm_dim)

        self.merge = nn.Linear(lstm_dim*2, embed_dim)

        self.predict = nn.Sequential(
            nn.Linear(embed_dim, len(labels)),
            nn.LogSoftmax(1),
        )

    def batches_iter(self, lines_iter, size=20):
        """Generate (lines, label idx) batches.
        """
        for lines in chunked_iter(lines_iter, size):

            # Labels -> indexes.
            yt_idx = [self.ltoi[line.label] for line in lines]
            yt = torch.LongTensor(yt_idx).type(itype)

            yield lines, yt

    def embed(self, lines):
        """Embed lines.
        """
        # Flat tokens.
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


class Trainer:

    def __init__(self, root, skim=None, lstm_dim=1024, embed_dim=512, lr=1e-4,
        batch_size=50, test_size=10000, eval_every=100000):

        self.corpus = Corpus(root, skim)

        labels = self.corpus.labels()

        token_counts = self.corpus.token_counts()

        self.model = Classifier(labels, token_counts, lstm_dim, embed_dim)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.batch_size = batch_size

        self.eval_every = eval_every

        self.train_lines, self.val_lines = train_test_split(
            self.corpus.lines, test_size=test_size)

        if torch.cuda.is_available():
            self.model.cuda()

    def train(self, epochs=10):
        """Train for N epochs.
        """
        for epoch in range(epochs):
            self.train_epoch(epoch)

    def train_epoch(self, epoch):

        logger.info('Epoch %d' % epoch)

        lines_iter = tqdm(self.train_lines)

        batches = self.model.batches_iter(lines_iter, self.batch_size)

        batch_losses = []
        eval_n = 0
        for lines, yt in batches:

            self.model.train()
            self.optimizer.zero_grad()

            yp = self.model(lines)

            loss = F.nll_loss(yp, yt)
            loss.backward()

            self.optimizer.step()

            batch_losses.append(loss.item())

            n = math.floor(lines_iter.n / self.eval_every)

            if n > eval_n:
                self.log_metrics(batch_losses)
                eval_n = n

        self.log_metrics(batch_losses)

    def log_val_metrics(self):

        self.model.eval()

        batches = self.model.batches_iter(tqdm(self.val_lines))

        yt, yp = [], []
        for lines, yti in batches:
            yp += self.model(lines).tolist()
            yt += yti.tolist()

        yt = torch.LongTensor(yt).type(itype)
        yp = torch.FloatTensor(yp).type(ftype)

        preds = yp.argmax(1)

        # LOSS
        loss = F.nll_loss(yp, yt)
        logger.info('Val loss: %f' % loss)

        # ACCURACY
        acc = metrics.accuracy_score(yt, preds)
        logger.info('Val acc: %f' % acc)

        yt_lb = [self.model.labels[i] for i in yt.tolist()]
        yp_lb = [self.model.labels[i] for i in preds.tolist()]

        # REPORT
        report = metrics.classification_report(yt_lb, yp_lb)
        logger.info('\n' + report)

    def log_metrics(self, train_losses):
        logger.info('Train loss: %f' % np.mean(train_losses))
        self.log_val_metrics()


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

    def lines_iter(self):
        """Generate encoded lines + metadata.
        """
        batches = self.model.batches_iter(tqdm(self.corpus), self.batch_size)

        for lines, yt in batches:

            embeds = self.model.embed(lines)
            embeds = embeds.cpu().detach().numpy()

            for line, embed in zip(lines, embeds):

                # Metadata + embedding.
                data = {**line.__dict__}
                data['embedding'] = embed

                yield data

    def segments_iter(self):
        """Generate (fname, data).
        """
        chunks = chunked_iter(self.lines_iter(), self.segment_size)

        for i, lines in enumerate(chunks):
            fname = '%s.p' % str(i).zfill(5)
            yield fname, pickle.dumps(lines)

    def write_fs(self, root):
        """Flush to local filesystem.
        """
        for fname, data in self.segments_iter():
            path = os.path.join(root, fname)
            write_fs(path, data)
