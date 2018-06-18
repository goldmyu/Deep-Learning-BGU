from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Implementation of section 3 in Ex2


tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model(features, labels, mode):
    # define the input
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # first conv layer with 5x5x32 dim
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # second layer batch-norm
    norm1 = tf.layers.batch_normalization(inputs=conv1)

    # third-layer max-pooling
    pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[2, 2], strides=2)

    # fourth layer of conv with 5x5x64 dim
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    # fifth batch-norm layer
    norm2 = tf.layers.batch_normalization(inputs=conv2)

    # sixth max-pooling layer
    pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

    # flatning the max-pool output for the following dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # seventh layer - dense of 1024
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # dropout of 40%
    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense2 = tf.layers.dense(inputs=dropout, units=1024, activation=tf.nn.relu)

    logits = tf.layers.dense(inputs=dense2, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # get mnist data using tf functionality
    mnist_data = tf.contrib.learn.datasets.load_dataset("mnist")

    # get mnist train data
    train_data = mnist_data.train.images
    train_labels = np.asarray(mnist_data.train.labels, dtype=np.int32)

    # get mnist test data
    test_data = mnist_data.test.images
    test_labels = np.asarray(mnist_data.test.labels, dtype=np.int32)

    # get mnist validation data
    validation_data = mnist_data.validation.images
    validation_labels = np.asarray(mnist_data.validation.labels, dtype=np.int32)

    run_config = tf.estimator.RunConfig(
        log_step_count_steps=250,
    )

    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model, config=run_config)


    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    mnist_classifier.train(input_fn=train_input_fn, steps=5000)

    # Evaluate the model using the test-set and print results
    eval_test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)

    eval_test_results = mnist_classifier.evaluate(input_fn=eval_test_input_fn)
    print("the accuracy on the test set is : %g" % eval_test_results["accuracy"])

    # Evaluate the model using the validation-set and print results
    eval_validate_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": validation_data},
        y=validation_labels,
        num_epochs=1,
        shuffle=False)

    eval_validate_results = mnist_classifier.evaluate(input_fn=eval_validate_input_fn)
    print("the accuracy on the validation set is : %g" % eval_validate_results["accuracy"])


if __name__ == "__main__":
    tf.app.run()
