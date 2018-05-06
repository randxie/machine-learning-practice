from collections import deque
import numpy as np

class TreeNode(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.weights = None
        self.split_point = None

    def add_split(self, feature_col, split_point):
        self.feature_col = feature_col
        self.split_point = split_point

    def find_direction(self, X):
        if self.weights is None:
            if X[self.feature_col] <= self.split_point:
                return self.left
            else:
                return self.right
        else:
            return self.weights


class MultiRegressionTree(object):
    def __init__(self, reg_lambda=1, depth=3):
        self.reg_lambda = reg_lambda
        self.depth = depth
        self.root = TreeNode()

    def fit(self, X, y):
        num_samples, num_cols = X.shape
        num_targets = y.shape[1]

        task_queue = deque()
        task_queue.append((np.arange(num_samples), self.root, 0))

        while len(task_queue) > 0:
            task = task_queue.popleft()
            instance_set = task[0]
            node = task[1]
            depth = task[2]

            if depth == self.depth:
                # compute weight
                node.weights = np.mean(y[instance_set, :], axis=0)
            else:
                Xi = X[instance_set, :]
                yi = y[instance_set, :]

                num_samples, num_cols = Xi.shape

                max_gain = 0
                best_split = 0
                best_col = 0

                for col in range(num_cols):
                    idx = np.argsort(Xi[:, col])

                    sum_y = np.sum(yi, axis=0)[:, None]
                    cumu_y = np.zeros((num_targets, 1))

                    for i in range(len(idx)):
                        cumu_y += yi[idx[i], :][:, None]
                        sum_y -= yi[idx[i], :][:, None]

                        gain_L = np.dot(cumu_y.T, cumu_y) / (i + 1)
                        gain_R = np.dot(sum_y.T, sum_y) / (num_samples - i + 1)

                        if (gain_L + gain_R) > max_gain:
                            best_split = Xi[idx[i], col]
                            best_col = col
                            max_gain = (gain_L + gain_R)

                left_instance_set = instance_set[Xi[:, best_col] <= best_split]
                right_instance_set = instance_set[Xi[:, best_col] > best_split]

                node.add_split(best_col, best_split)
                node.left = TreeNode()
                node.right = TreeNode()

                task_queue.append((left_instance_set, node.left, depth+1))
                task_queue.append((right_instance_set, node.right, depth+1))


    def predict(self, X):
        node = self.root
        while isinstance(node, TreeNode):
            # find the weight vector
            node = node.find_direction(X)

        ret = node
        return ret


X = np.random.randn(1000, 3)
y1 = (X[:,0] > 0.5)[:, None]
y2 = (X[:,1] < 0.1)[:, None]
y = np.hstack((y1, y2)).astype(np.int)

mdl  = MultiRegressionTree(1, 2)
mdl.fit(X, y)

for i in range(10):
    print('-------------------')
    print('Predicting sample: %d'%(i))
    print(mdl.predict(X[0, :]))
    print(y[0, :])

mse = 0
for i in range(X.shape[0]):
    dy = (mdl.predict(X[i, :]) - y[i, :])
    mse += np.sum(dy * dy)

mse = np.sqrt(mse/X.shape[0])
print('mean square error: %f'%(mse))

print('Splitting point')

# bfs by level
level = 0
traverser = deque()
traverser.append(mdl.root)
while len(traverser) > 0:
    print('At level: %d'%(level))
    level += 1
    to_traverse = len(traverser)
    for i in range(to_traverse):
        node = traverser.popleft()
        if node.split_point is not None:
            print('(%d, %f) '%(node.feature_col, node.split_point))
            traverser.append(node.left)
            traverser.append(node.right)