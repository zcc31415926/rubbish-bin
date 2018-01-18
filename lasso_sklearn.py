import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import mglearn

X,y=mglearn.datasets.load_extended_boston()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
lasso=Lasso().fit(X_train,y_train)

print(format(lasso.score(X_train,y_train)))
print(format(lasso.score(X_test,y_test)))
print(format(np.sum(lasso.coef_!=0)))
