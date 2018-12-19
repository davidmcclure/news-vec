

import sys


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
