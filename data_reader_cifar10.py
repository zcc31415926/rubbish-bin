import pickle
import numpy as np
import cv2


def unpickle(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    # keys: batch_label, labels, data, filenames
    return data_dict


def load_dataset(data_path):
    train_data = []
    train_labels = []
    # train batches
    for i in range(1, 6):
        filename = data_path + 'data_batch_' + str(i)
        data_dict = unpickle(filename)
        train_data.append(data_dict[b'data'])
        train_labels.append(data_dict[b'labels'])
    train_data = np.concatenate(train_data, 0)
    train_labels = np.concatenate(train_labels, 0)
    # test batches
    filename = data_path + 'test_batch'
    data_dict = unpickle(filename)
    val_data, val_labels = data_dict[b'data'], data_dict[b'labels']
    train_data = np.reshape(train_data, (50000, 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0
    val_data = np.reshape(val_data, (10000, 3, 32, 32)).transpose(0, 2, 3, 1) / 255.0
    return [train_data, train_labels], [val_data, val_labels]


if __name__ == '__main__':
    data_path = './Data/cifar10/'
    train_set, test_set = load_dataset(data_path)
    print(np.shape(train_set[0]))
    print(np.shape(train_set[1]))
    print(np.shape(test_set[0]))
    print(np.shape(test_set[1]))
    img = train_set[0][0]
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

