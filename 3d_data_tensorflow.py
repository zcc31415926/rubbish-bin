import tensorflow as tf
import numpy as np

# generate 100 data points
x_data=np.float32(np.random.rand(2,100))
y_data=np.dot([0.100,0.200],x_data)+0.300

# generate a linear model
b=tf.Variable(tf.zeros([1]))
W=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y=tf.matmul(W,x_data)+b

# minimize variance
loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

# initialize all variables
init=tf.global_variables_initializer()

# start graph
sess=tf.Session()
sess.run(init)

# plane fitting
for step in xrange(0,201):
    sess.run(train)
    if step%20==0:
        print (step,sess.run(W),sess.run(b))

# get the best fitting result
