import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# Load mnist data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# place holder for input data X
x = tf.placeholder(tf.float32, [None, 784])

# place-holder for the wights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# place holder for the approximated Y_hat
y = tf.nn.softmax(tf.matmul(x, W) + b)


# the actual label data Y
y_ = tf.placeholder(tf.float32, [None, 10])

# compute the cross-entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# train phase - 0.5 is alpha
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# lunch the session after all the config
sess = tf.InteractiveSession()

# init all the variables
tf.global_variables_initializer().run()

# train in batches of 100 - getting 100 !!!RANDOM!!! data point
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
