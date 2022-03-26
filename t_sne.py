# based on zhenguo's code

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE


def get_data():
    digits = datasets.load_digits(n_class=10)
    data = digits.data
    label = digits.target
    num_samples, dimension = data.shape # 1797*64
    return data, label, num_samples, dimension

def plot_embedding(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 10),
                 fontdict={'weight': 'bold', 'size': 10})
    return fig


if __name__ == '__main__':
    data, label, num_samples, dimension = get_data()
    ts = TSNE(n_components=2, init='pca', random_state=0)
    result = ts.fit_transform(data)
    fig = plot_embedding(result, label)
    plt.savefig('t_sne_result.png')

