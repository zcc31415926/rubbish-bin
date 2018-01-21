import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.linear_model import LinearRegression
X,y=mglearn.datasets.make_wave(n_samples=60)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

lr=LinearRegression().fit(X_train,y_train)
print(format(lr.coef_))
print(format(lr.intercept_))

# determine the accurate rate of the prediction
# training set score
print(format(lr.score(X_train,y_train)))
# test set score
print(format(lr.score(X_test,y_test)))
