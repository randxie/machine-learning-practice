
class DisjointSet(object):
    """
    As described in https://www.cs.princeton.edu/~rs/AlgsDS07/01UnionFind.pdf
    """
    def __init__(self, N: int):
        self.ids = [i for i in range(N)]
        self.capacity = [1] * N

    def find(self, p: int, q: int) -> bool:
        p = self._find_root(p)
        q = self._find_root(q)

        return p == q

    def union(self, p: int , q: int) -> None:
        # find root of nodes
        p_root = self._find_root(p)
        q_root = self._find_root(q)

        if p_root != q_root:
            # find corresponding capacity
            c_p = self.capacity[p_root]
            c_q = self.capacity[q_root]

            if c_p < c_q:
                # attach q to p by swapping
                p_root, q_root = q_root, p_root

            self.ids[q_root] = p_root
            self.capacity[p_root] += self.capacity[q_root]

    def _find_root(self, p: int) -> int:
        while p != self.ids[p]:
            p = self.ids[p]

        return p


if __name__ == '__main__':
    edges = [(3, 4), (4, 9), (8, 0), (2, 3),
             (5, 6), (5, 9), (7, 3), (4, 8),
             (6, 1)]

    mdl = DisjointSet(N=10)

    for e in edges:
        mdl.union(e[0], e[1])

    print(mdl.ids)
