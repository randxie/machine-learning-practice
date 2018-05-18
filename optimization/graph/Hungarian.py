import numpy as np

class HungarianLinearMatching(object):
    """
    Follow description in
    [1] https://en.wikipedia.org/wiki/Hungarian_algorithm#Matrix_interpretation
    [2] https://www.geeksforgeeks.org/hungarian-algorithm-assignment-problem-set-1-introduction/
    """

    def __init__(self, mtx):
        self.mtx = mtx

    def row_reduction(self):
        row_min = np.amin(self.mtx, axis=1)
        self.mtx -= row_min[:, None]

    def col_reduction(self):
        col_min = np.amin(self.mtx, axis=0)
        self.mtx -= col_min

    def find_minimum_cover(self):
        # assign tasks
        N = self.mtx.shape[0]
        assign_mtx = (self.mtx == 0)
        is_assigned = np.zeros((N, 1))
        for i in range(N):
            indices = np.where(assign_mtx[i, :] > 0)[0]
            if len(indices) > 0:
                cur_idx = indices[0]
                assign_mtx[:, cur_idx] = False
                is_assigned[i] = 1

        # drawing lines
        marked_row = (1 - is_assigned)
        marked_col = np.zeros((N, 1))
        assign_mtx = (self.mtx == 0)

        for i in range(N):
            if marked_row[i]:
                to_mark_col = np.where(assign_mtx[i, :] > 0)[0]
                if len(to_mark_col) > 0:
                    marked_col[to_mark_col] = 1
                    for j in range(len(to_mark_col)):
                        to_mark_row = np.where(assign_mtx[:, to_mark_col[j]])[0]
                        marked_row[to_mark_row] = 1

        N_cover = np.sum(marked_col) + np.sum(1 - marked_row)
        cover_row = (1 - marked_row)
        cover_col = marked_col

        return N_cover, np.ravel(cover_row), np.ravel(cover_col)

    def rebalance_entries(self, cover_row, cover_col):
        uncover_mtx = np.ones(self.mtx.shape)
        uncover_mtx[cover_row == 1, :] = 0
        uncover_mtx[:, cover_col == 1] = 0
        min_entry = np.min(self.mtx[uncover_mtx == 1])

        self.mtx[cover_row == 0, :] -= min_entry
        self.mtx[:, cover_col == 1] += min_entry

    def get_optimal_match(self):
        N = self.mtx.shape[0]
        assign_mtx = (self.mtx == 0)
        solution = []
        for i in range(N):
            indices = np.where(assign_mtx[i, :])[0]
            if len(indices) > 0:
                cur_idx = indices[0]
                assign_mtx[:, cur_idx] = 0
                solution.append((i, cur_idx))

        return solution

    def compute(self):
        N = self.mtx.shape[0]
        self.row_reduction()
        self.col_reduction()

        N_cover, cover_row, cover_col = self.find_minimum_cover()

        while N_cover < N:
            self.rebalance_entries(cover_row, cover_col)
            N_cover, cover_row, cover_col = self.find_minimum_cover()

        return self.mtx


if __name__ == '__main__':

    # test case 1
    match_mtx = np.array([[40, 60, 15], [25, 30, 45], [55, 30, 25]])

    mdl = HungarianLinearMatching(match_mtx)

    print('----- Match matrix --------')
    print(match_mtx)

    print('----- Final matrix --------')
    print(mdl.compute())

    print('----- Optimal matching --------')
    print(mdl.get_optimal_match())

