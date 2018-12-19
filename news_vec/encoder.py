

import os
import pickle

from torch.utils.data import DataLoader
from boltons.iterutils import chunked_iter

from . import logger, utils


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
        loader = DataLoader(
            self.corpus,
            collate_fn=self.model.collate_batch,
            batch_size=self.batch_size,
        )

        for i, (lines, yt) in enumerate(loader):

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
                    **line.__dict__,
                    **preds,
                    embedding=embed,
                )

                yield data

            utils.print_replace(loader.batch_size * (i+1))

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
