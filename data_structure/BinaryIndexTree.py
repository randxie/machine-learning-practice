import numpy as np


class BinaryIndexTree(object):
    """ Also known as Fenwick Tree
    Refer to
    [1] https://activities.tjhsst.edu/sct/lectures/1213/bit_09_28_12.pdf
    [2] https://www.geeksforgeeks.org/binary-indexed-tree-or-fenwick-tree-2/
    """
    def __init__(self, data: list):
        self.psum = [0] * (len(data) + 1)

        for idx, d in enumerate(data):
            self.add(idx+1, d)

    def lowbit(self, x: int):
        return x & (-x)

    def add(self, x: int, d):
        while x < len(self.psum):
            self.psum[x] += d
            x += self.lowbit(x)

    def sum(self, x: int):
        ret = 0
        while x > 0:
            ret += self.psum[x]
            x -= self.lowbit(x)

        return ret


if __name__ == '__main__':
    # test case 1
    data = [1, -1, 2, -2, 3, -3, 4, -4]
    mdl = BinaryIndexTree(data)

    print(np.cumsum(data))
    print(np.array([mdl.sum(i+1) for i in range(len(data))]))

    # test case 2
    data = [1, 2, 3, 4, -4, -3, -2, -1]
    mdl = BinaryIndexTree(data)

    print(np.cumsum(data))
    print(np.array([mdl.sum(i+1) for i in range(len(data))]))