import tensorflow as tf

# a sample session

# generate two constant() operations
matrix1=tf.constant([[3.,3.]])
matrix2=tf.constant([[2.],[2.]])

# generate one matmul() operations
product=tf.matmul(matrix1,matrix2)

# start default graph
sess=tf.Session()
result=sess.run(product)
print(result)

# close the session to release the resources
sess.close()
# autoclose with 'with'
# with tf.Session() as sess:
#     result=sess.run([product])
#     print result

# a sample interactive session

intersess=tf.InteractiveSession()

# generate a variable and a constant
x=tf.Variable([1.0,2.0])
a=tf.constant([3.0,3.0])

# initialize variable x
x.initializer.run()

# add a 'sub' operation
# 'subtract' in new versions of API
sub=tf.subtract(x,a)
print(sub.eval())
