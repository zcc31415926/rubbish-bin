# copied from tensorflow manual v1.2

# includes
import tensorflow as tf


# initialize weight assignment
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)


# initialize bias assignment
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


# convolutional layer with stride 1 and padding 0
def conv2d_1x0(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')


# pooling layer with 2*2 kernel size
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


x_image = tf.reshape(x, [-1, 28, 28, 1])
sess = tf.Session()
saver = tf.train.Saver()

# the first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# the first pooling layer
h_conv1 = tf.nn.relu(conv2d_1x0(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# the second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# the second pooling layer
h_conv2 = tf.nn.relu(conv2d_1x0(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# the first fully-connected layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# spread the 64 7*7 pics into a 7*7*64 vector
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout layer
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# the second fully_connected layer (softmax layer)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# judgement: cross entropy
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# run the session

sess.run(tf.global_variables_initializer())

# for i in range(20000):
#     batch = mnist.train.next_batch(50)
#     if i%100 == 0:
#         train_accuracy = accuracy.eval(feed_dict = {x: batch[0],
#                                                     y_: batch[1],
#                                                     keep_prob: 1.0})
#         print("step %d, training accuracy %g"%(i, train_accuracy))
#         # save the present state and parameters as a checkpoint
#         # stored in "model.ckpt"
#         saver.save(sess, 'model.ckpt', global_step = i+1)
#     train_step.run(feed_dict = {x: batch[0],
#                                 y_: batch[1],
#                                 keep_prob = 0.5})
#
# print("test accuracy %g"%accuracy.eval(feed_dict = {x: mnist.test.images,
#                                                     y_: mnist.test.labels,
#                                                     keep_prob: 1.0}))

# # restore the checkpoint state
# checkpoint = tf.train.get_checkpoint_state('')
# saver.restore(sess, checkpoint.model_checkpoint_path)
# # print variables in the model
# print(sess.run(W_conv1))

sess.close()
