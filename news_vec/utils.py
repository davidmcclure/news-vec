

import sys

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


class ProgressDataLoader(DataLoader):

    def __iter__(self):
        """Track # generated pairs.
        """
        self.n = 0

        for x, y in super().__iter__():
            self.n += len(x)
            print_replace(f'{self.n}/{len(self.dataset)}\r')
            yield x, y
