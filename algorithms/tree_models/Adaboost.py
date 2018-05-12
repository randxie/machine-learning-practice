import numpy as np
import matplotlib.pyplot as plt


class DecisionStump():
    def __init__(self):
        self.col_idx = -1
        self.direction = 0
        self.val = 0

    def _get_loss(self, y, ypred, weight):
        ''' use mean squared error '''
        loss = np.dot((y != ypred), weight)
        return loss

    def _predict(self, vec, val, direction):
        return (2 * ((vec >= val) - 0.5) * direction)

    def get_error(self, data_in, data_out, weight):
        ypred = self.predict(data_in)
        error = np.dot((data_out != ypred), weight)
        return error

    def predict(self, data_in):
        tmp_data = data_in[:, self.col_idx]
        ypred = self._predict(tmp_data, self.val, self.direction)
        return ypred

    def find_split(self, vec_in, vec_out, weight):
        order = np.argsort(vec_in)
        vec_in = vec_in[order]
        weight = weight[order]
        vec_out = vec_out[order]

        # find best decision
        minLoss = len(vec_out)
        optimalDir = 0
        optimalVal = 0
        for i in range(len(vec_in)):
            decision = vec_in[i]
            for direction in [-1, 1]:
                loss = self._get_loss(vec_out, self._predict(vec_in, decision, direction), weight)
                if (loss < minLoss):
                    optimalDir = direction
                    optimalVal = decision
                    minLoss = loss
        return optimalVal, optimalDir, minLoss

    def train(self, data_in, data_out, weight):
        num_col = data_in.shape[1]
        minLoss = len(data_out)
        for i in range(num_col):
            vec_in = data_in[:, i]
            val, direction, loss = self.find_split(vec_in, data_out, weight)
            if (loss < minLoss):
                self.col_idx = i
                self.val = val
                self.direction = direction
                minLoss = loss

class AdaBoost:
    def __init__(self):
        self.max_iter = 3
        self.stumps = []
        self.alpha = []

    def fit(self, data_in, data_out):
        weight = np.ones((data_in.shape[0], 1)) / len(data_out)
        for m in range(self.max_iter):
            stump = DecisionStump()
            stump.train(data_in, data_out, weight)
            err_m = stump.get_error(data_in, data_out, weight)
            alpha_m = 1 / 2 * np.log((1 - err_m) / err_m) if err_m != 0 else 0
            idxWrong = (data_out != stump.predict(data_in))
            weight[idxWrong] = weight[idxWrong] * np.exp(alpha_m)
            weight = weight / np.sum(weight)
            self.alpha.append(alpha_m)
            self.stumps.append(stump)

    def predict(self, data_in):
        # check that data is a matrix (columns = features; rows = entries)
        if not isinstance(data_in, np.ndarray):
            data_in = np.array(data_in)

        # build return array
        ypred = np.zeros((data_in.shape[0], 1))

        for i in range(len(self.stumps)):
            ypred += self.alpha[i] * np.reshape(self.stumps[i].predict(data_in), (len(ypred), 1))

        return np.sign(ypred)


if __name__ == '__main__':
    data_in = np.array([(1, 1.2), (0.8, 2), (2, 2.7), (2.4, 2.9), (3, 2.5), (1.4, 0.6),
                        (1.6, 2.1), (2.5, 2.1), (3.5, 1.2), (3.6, 2.8)])
    data_out = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

    # fit and predict
    mdl = AdaBoost()
    mdl.fit(data_in, data_out)
    y_pred = mdl.predict(data_in)[:, 0]

    # visualize results
    plt.scatter(data_in[data_out == -1, 0], data_in[data_out == -1, 1], c='b', s=40, label='True Good')
    plt.scatter(data_in[data_out == 1, 0], data_in[data_out == 1, 1], c='r', s=40, label='True Bad')
    plt.scatter(data_in[y_pred == -1, 0], data_in[y_pred == -1, 1], c='b', s=80, marker='x', label='Predict Good')
    plt.scatter(data_in[y_pred == 1, 0], data_in[y_pred == 1, 1], c='r', s=80, marker='x', label='Predict Bad')
    plt.legend()
    plt.show()
