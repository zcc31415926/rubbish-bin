from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split\
(cancer.data,cancer.target,stratify=cancer.target,random_state=42)
logreg=LogisticRegression().fit(X_train,y_train)
print('C=1')
print(format(logreg.score(X_train,y_train)))
print(format(logreg.score(X_test,y_test)))
print

print('C=100')
logreg100=LogisticRegression(C=100).fit(X_train,y_train)
print(format(logreg100.score(X_train,y_train)))
print(format(logreg100.score(X_test,y_test)))
print

print('C=0.01')
logreg001=LogisticRegression(C=0.01).fit(X_train,y_train)
print(format(logreg001.score(X_train,y_train)))
print(format(logreg001.score(X_test,y_test)))
print

# the parameter C is the regularization strength
# stronger regularization pushes coefficients more toward zero
# larger C results in more accurate scores
