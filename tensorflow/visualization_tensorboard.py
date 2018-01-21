# copied from tensorflow implementations for machine intelligence

# includes
import tensorflow as tf

# constants and operators, as nodes in the graph session
a = tf.constant(5, name = 'input-a')
b = tf.constant(3, name = 'input-b')
c = tf.multiply(a, b, name = 'mul-c')
d = tf.add(a, b, name = 'add-d')
e = tf.add(c, d, name = 'add-e')

# run the session
sess = tf.Session()
output = sess.run(e)

# record the process
writer = tf.summary.FileWriter('./my_graph', sess.graph)

# close the summary writer and the graph session
writer.close()
sess.close()
