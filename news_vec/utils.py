

import sys
import ujson
import gzip

from glob import glob
from torch.utils.data import DataLoader


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


def tensor_to_np(tensor):
    return tensor.cpu().detach().numpy()


def print_replace(msg):
    sys.stdout.write(f'\r{msg}')
    sys.stdout.flush()


def read_json_gz_lines(root):
    """Read JSON corpus.

    Yields: dict
    """
    for path in glob('%s/*.gz' % root):
        with gzip.open(path) as fh:
            for line in fh:
                yield ujson.loads(line)


class ProgressDataLoader(DataLoader):

    def __iter__(self):
        """Track # generated pairs.
        """
        self.n = 0

        for x, y in super().__iter__():
            self.n += len(x)
            print_replace(f'{self.n}/{len(self.dataset)}\r')
            yield x, y
