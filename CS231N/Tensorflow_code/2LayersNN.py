import tensorflow as tf
import numpy as np

# Initialise the inputs, outputs and weights
N, D, H = 64, 1000, 100
x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.Variable(tf.random_normal((D, H)))
W2 = tf.Variable(tf.random_normal((H, D)))

# Forward pass
h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y
loss = tf.reduce_mean(tf.reduce_sum(diff * diff), axis=1)

optimiser = tf.train.GradientDescentOptimizer(1e-5)
updates = optimiser.minimize(loss)

with tf.Session() as sess:
	sedd.run(tf.global_variables_initializer())