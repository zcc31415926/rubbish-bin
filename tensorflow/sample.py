# test program

# includes
import tensorflow as tf

a = tf.Variable(1, name = 'var_a')
init = tf.initialize_variables([a], name = 'init_a')
