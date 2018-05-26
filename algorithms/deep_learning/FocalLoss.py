import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def focal_loss(labels, predictions, gamma=0):
    """
    Only for binary case
    """
    pt = tf.multiply(labels, predictions) + tf.multiply(1 - labels, 1 - predictions)
    fl = -tf.multiply(tf.pow(1-pt, gamma), tf.log(tf.clip_by_value(pt, 1e-16, 1)))
    return fl


if __name__ == '__main__':
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    gamma = tf.placeholder(tf.float32)

    loss_op = focal_loss(y, x, gamma)

    with tf.Session():
        for gm in [0, 0.5, 1, 2, 5]:
            x_data = np.linspace(0.01, 1, 100)
            y_data = np.ones(len(x_data))
            loss = loss_op.eval(feed_dict={x: x_data, y: y_data, gamma: gm})
            plt.plot(x_data, loss, label='Gamma=%0.2f' % gm)

    plt.legend()
    plt.show()