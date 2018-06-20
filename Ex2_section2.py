# Section 2 in Ex2
#
# implementing the network :
#
# Conv1_5x5x32_same --> Maxpool_2x2_Stride: 2 --> Conv2_5x5x64_same -->
# Maxpool_2x2_Stride: 2 --> Dense1_Size:1024 --> Dense2_Size:1024 --> Softmax

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

# ------------- utility functions ---------------------


def create_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def create_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# ------------------ configure the network ---------------------------

# get the MNIST data
mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

# init the tf session
sess = tf.InteractiveSession()

# place-holders for input data X and labeled data Y
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# -------------- build the network architecture -------------------


# first conv layer 5x5x32 and maxPool 2x2 with stride 2
W_conv1 = create_weight_variable([5, 5, 1, 32])
b_conv1 = create_bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second conv layer 5x5x64 and maxPool of 2x2 with stride 2
W_conv2 = create_weight_variable([5, 5, 32, 64])
b_conv2 = create_bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# third layer - dense FC of 1024
W_fc1 = create_weight_variable([7 * 7 * 64, 1024])
b_fc1 = create_bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# fourth layer - another dense FC layer of 1024
W_fc2 = create_weight_variable([1024, 1024])
b_fc2 = create_bias_variable([1024])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

# fifth layer - softmax output
W_fc3 = create_weight_variable([1024, 10])
b_fc3 = create_bias_variable([10])

y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

# ---------------------- train and evaluate ------------------------------

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())


for i in range(5000):
    batch = mnist.train.next_batch(100)

    if i % 250 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        cross_entropy_loss = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]})
        print("step %d, the training loss is : %g and training accuracy is : %g" % (i,cross_entropy_loss, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print("validation accuracy %g" % accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))
