from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons,load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

# an example of random forest
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split\
(cancer.data,cancer.target,random_state=0)

forest=RandomForestClassifier(n_estimators=100,random_state=0)
forest.fit(X_train,y_train)

print(format(forest.score(X_train,y_train)))
print(format(forest.score(X_test,y_test)))
print

# another example
X,y=make_moons(n_samples=100,noise=0.25,random_state=3)
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42)

forest=RandomForestClassifier(n_estimators=5,random_state=2)
forest.fit(X_train,y_train)

fig,axes=plt.subplots(2,3,figsize=(20,10))
for i,(ax,tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
    ax.set_title(format(i))
    mglearn.plots.plot_tree_partition(X_train,y_train,tree,ax=ax)

mglearn.plots.plot_2d_separator\
(forest,X_train,fill=True,ax=axes[-1,-1],alpha=0.4)
axes[-1,-1].set_title('random_forest')
mglearn.discrete_scatter(X_train[:,0],X_train[:,1],y_train)

# for n_estimators, larger is always better.
# but need more memory and more time to train.
