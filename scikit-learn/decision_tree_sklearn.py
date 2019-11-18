import graphviz
import mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import export_graphviz


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print('unlimited depth: ')
print(format(tree.score(X_train, y_train)))
print(format(tree.score(X_test, y_test)))

# sample decision tree
# a leaf of the tree that contains data points
# that all share the same target value
# is called pure
# pure leaves result in 100% accuracy on the training set

# limit the depth to 4:
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print('max_depth = 4: ')
print(format(tree.score(X_train, y_train)))
print(format(tree.score(X_test, y_test)))

export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"], feature_names=cancer.feature_names, impurity=False, filled=True)
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

# determine the importance of each feature
print(format(tree.feature_importances_))

# histograms are more intuitionistic
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature importance")
    plt.ylabel("feature")

plot_feature_importances_cancer(tree)
plt.show()

# another sample
another_tree = mglearn.plots.plot_tree_not_monotone()
display(tree)
