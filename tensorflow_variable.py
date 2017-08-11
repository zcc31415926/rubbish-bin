import tensorflow as tf

state=tf.Variable(0,name='counter')
one=tf.constant(1)
new_value=tf.add(state,one)
update=tf.assign(state,new_value)

# variables must be initialized
init_op=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    for i in range(3):
        sess.run(update)
        print(sess.run(state))

# get tensors (multiple results) at one run
input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)
intermed=tf.add(input2,input3)
mul=tf.multiply(input1,intermed)

with tf.Session() as sess:
    result=sess.run([mul,intermed])
    print(result)
