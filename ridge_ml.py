# ridge regularization is to explicitly restrict the model to avoid overfitting
# use ridge to avoid the huge difference
# between train set score and test set score

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.linear_model import Ridge
X,y=mglearn.datasets.make_wave(n_samples=60)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

ridge=Ridge().fit(X_train,y_train)

# determine the accurate rate of the prediction
print("alpha=1")
# training set score
print(format(ridge.score(X_train,y_train)))
# test set score
print(format(ridge.score(X_test,y_test)))
print

# the alpha parameter affects the scores
ridge10=Ridge(alpha=10).fit(X_train,y_train)
print("alpha=10")
print(format(ridge10.score(X_train,y_train)))
print(format(ridge10.score(X_test,y_test)))
print

ridge01=Ridge(alpha=0.1).fit(X_train,y_train)
print("alpha=0.1")
print(format(ridge01.score(X_train,y_train)))
print(format(ridge01.score(X_test,y_test)))
print
