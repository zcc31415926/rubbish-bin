from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def read_MNIST_dataset(data_path):
    mnist = input_data.read_data_sets(data_path, one_hot=False)
    return mnist.train.images, mnist.train.labels, \
        mnist.validation.images, mnist.validation.labels, \
        mnist.test.images, mnist.test.labels

if __name__ == '__main__':
    import cv2

    data_path = '/home/charlie/Data/MNIST/'
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = \
        read_MNIST_dataset(data_path)
    print(np.shape(train_images))
    print(np.shape(train_labels))
    print(np.shape(validation_images))
    print(np.shape(validation_labels))
    print(np.shape(test_images))
    print(np.shape(test_labels))
    img = train_images[0]
    label = train_labels[0]
    img = np.reshape(img, (28, 28))
    print(label)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

