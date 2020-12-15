import tensorflow as tf
from tensorflow.contrib.slim import nets
import cv2
import numpy as np

img = cv2.imread('./street.jpeg') / 255.0
img = cv2.resize(img, (224, 224), cv2.INTER_CUBIC)
img = np.expand_dims(img, 0)
img = tf.convert_to_tensor(img, dtype=tf.float32)

with tf.Session() as sess:
    with tf.GradientTape() as g:
        g.watch(img)
        output, _ = nets.resnet_v1.resnet_v1_50(img, num_classes=10, is_training=True)
        grad = g.gradient(output, img)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        grad_value = sess.run(grad)
        print('gradients:', grad_value)

