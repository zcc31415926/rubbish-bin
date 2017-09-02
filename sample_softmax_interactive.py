import input_data
import tensorflow as tf

mnist=input_data.read_data_sets("/home/charlie/Documents/MNIST/",one_hot=True)
sess=tf.InteractiveSession()
x=tf.placeholder("float32", shape=[None,784])
y0=tf.placeholder("float32", shape=[None,10])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

y=tf.nn.softmax(tf.matmul(x,W)+b)
cross_entropy=-tf.reduce_sum(y0*tf.log(y))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(1000):
    batch=mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], y0:batch[1]})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y0,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float32"))
print(accuracy.eval(feed_dict={x:mnist.test.images,y0:mnist.test.labels}))
