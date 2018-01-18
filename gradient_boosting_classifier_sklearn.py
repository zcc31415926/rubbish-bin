from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer=load_breast_cancer()

X_train,X_test,y_train,y_test=train_test_split\
(cancer.data,cancer.target,random_state=0)

# unlimited depth
gbrt=GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train,y_train)

print('unlimited depth')
print(format(gbrt.score(X_train,y_train)))
print(format(gbrt.score(X_test,y_test)))
print

# max_depth=1
gbrt=GradientBoostingClassifier(random_state=0,max_depth=1)
gbrt.fit(X_train,y_train)

print('max_depth=1')
print(format(gbrt.score(X_train,y_train)))
print(format(gbrt.score(X_test,y_test)))
print

# learning_rate=0.01
gbrt=GradientBoostingClassifier(random_state=0,learning_rate=0.01)
gbrt.fit(X_train,y_train)

print('learning_rate=0.01')
print(format(gbrt.score(X_train,y_train)))
print(format(gbrt.score(X_test,y_test)))
print

# limit max_depth and decrease learning_rate
# both methods of decreasing the model complexity
# reduce the training set accuracy

# parameters are hard to train
