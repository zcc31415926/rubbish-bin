from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ram_prices=pd.read_csv\
('/home/charlie/Documents/Python/'\
+'introduction_to_ml_with_python-master/data/ram_price.csv')

# use historical data to forecast prices after the year 2000
data_train=ram_prices[ram_prices.date<2000]
data_test=ram_prices[ram_prices>=2000]

# predict prices based on date
X_train=data_train.date[:,np.newaxis]
# log-transform to get a simpler relationship of data to target
y_train=np.log(data_train.price)

tree=DecisionTreeRegressor().fit(X_train,y_train)
linear_reg=LinearRegression().fit(X_train,y_train)

# predict on all data
X_all=ram_prices.date[:,np.newaxis]

pred_tree=tree.predict(X_all)
pred_lr=linear_reg.predict(X_all)

# undo log-transform
price_tree=np.exp(pred_tree)
price_lr=np.exp(pred_lr)

plt.semilogy(data_train.date,data_train.price,label='training data')
plt.semilogy(data_test.date,data_test.price,label='test data')
plt.semilogy(ram_prices.date,price_tree,label='tree prediction')
plt.semilogy(ram_prices.date,price_lr,label='linear prediction')
plt.legend()
plt.show()

# advantages of decision tree:
# the resulting model can easily be visualized
# the algorithms are completely invariant to scaling of the data

# disadvantages of decision tree:
# overfit and provide poor generalization performance
# setting either max_depth, max_leaf_nodes, or min_samples_leaf
# is sufficient to prevent overfitting
