# input_data.py is a tool python source file

# input_data.py starts
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import gzip
# import os
# import tempfile
# import numpy
# from six.moves import urllib
# from six.moves import xrange  # pylint: disable=redefined-builtin
# import tensorflow as tf
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
# input_data.py ends

import input_data
import tensorflow as tf

# read the data (in 'ubyte' format)
mnist=input_data.read_data_sets("/home/charlie/Documents/MNIST/",one_hot=True)

# divide each picture into 28*28 pixels
# each picture is represented by a 10*1 vector
x=tf.placeholder("float32",[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

# use softmax model
y=tf.nn.softmax(tf.matmul(x,W)+b)
# the actual value
y0=tf.placeholder("float32",[None,10])

# the training model starts

# calculate the cross entropy
cross_entropy=-tf.reduce_sum(y0*tf.log(y))
# use the gradient descent algorithm to minimize cross entropy
# 0.01 as the learning speed
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# the training model ends
# the training process starts

# initialization
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

# training for 1000 rounds
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y0:batch_ys})

# evaluate the training results
# roughly 91% accurate
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y0,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float32"))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y0:mnist.test.labels}))
