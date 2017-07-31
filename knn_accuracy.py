import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(cancer.data,
cancer.target,stratify=cancer.target,random_state=66)
from sklearn.neighbors import KNeighborsClassifier

training_accuracy=[]
test_accuracy=[]
# try n_neighbors from 1 to 10
neighbors_settings=range(1,11)

for n_neighbors in neighbors_settings:
    # build the model
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train,y_train)
    # record training set accuracy
    training_accuracy.append(clf.score(X_train,y_train))
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(neighbors_settings,training_accuracy,label="training_accuracy")
plt.plot(neighbors_settings,test_accuracy,label="test_accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
