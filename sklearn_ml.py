import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.datasets import load_iris
iris_dataset=load_iris()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris.dataset['data'],
iris_dataset['target'],random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

# determine the accurate rate of the prediction
knn.fit(X_train,y_train)
print(format(knn.score(X_test,y_test)))

# make a prediction
prediction=knn.predict(X_new)
print(format(prediction))
print(format(iris_dataset['target_names'][prediction]))
