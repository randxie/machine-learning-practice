import numpy as np
from functools import reduce
from collections import Counter


class Apriori(object):
    """
    Naive implementation in https://en.wikipedia.org/wiki/Apriori_algorithm
    Could be accelerated by hash tree.
    """
    def __init__(self, support: int = 3):
        self.frequent_items = []
        self.support = support

    def fit(self, data: list):
        self.frequent_items = []

        total_set = reduce(lambda x, y: x | y, data)
        Lk = [{x} for x in total_set]

        while len(Lk) > 0:
            L_cur = Counter()

            for i, s in enumerate(data):
                for l in Lk:
                    if l.issubset(s):
                        L_cur[str(l)] += 1

            L_cur = [x for x in L_cur if L_cur[x] >= self.support]
            if len(L_cur) > 0:
                self.frequent_items.append(L_cur.copy())

            # update candidate set
            Lk = []
            for l in L_cur:
                tmp = eval(l)
                Lk.extend([tmp.union({x}) for x in total_set if x > max(tmp)])


if __name__ == '__main__':
    data = [{1, 2, 3, 4},
            {1, 2, 4},
            {1, 2},
            {2, 3, 4},
            {2, 3},
            {3, 4},
            {2, 4}]

    mdl = Apriori(support=3)
    mdl.fit(data)

    print(mdl.frequent_items)
