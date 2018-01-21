# copied from tensorflow implementations for machine intelligence

# includes
import tensorflow as tf

# define two different graphs
g1 = tf.Graph()
g2 = tf.Graph()

# add operators to g1
with g1.as_default():
    a = tf.add(3, 5, name = 'add-a')
    b = tf.add(4, 6, name = 'add-b')

# add operators to g2
with g2.as_default():
    c = tf.multiply(3, 5, name = 'mul-c')
    d = tf.multiply(4, 6, name = 'mul-d')

# run g1 in session
with tf.Session(graph = g1) as sess:
    print(sess.run(a))

# run g2 in session
with tf.Session(graph = g2) as sess:
    print(sess.run(c))

# with the 'with' structure,
# it can automatically close the session after the process