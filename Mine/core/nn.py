import numpy as np
import tensorflow as tf


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(
        tf.truncated_normal([in_dim, out_dim],
                            stddev=xavier_stddev,
                            dtype=tf.float32),
        dtype=tf.float(32))


def initialize_nn(layers):
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(num_layers - 1):
        W = xavier_init([layers[l], layers[l + 1]])
        b = tf.Variable(
            tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
    return weights, biases


def neural_net(X, weights, biases, activation=tf.nn.relu):
    num_layers = len(weights) + 1
    H = X
    for l in range(num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = activation(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y
