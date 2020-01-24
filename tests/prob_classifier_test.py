from sklearn import svm, datasets
import numpy as np
from random import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(color_codes=True)

'''
Probabilistic Initiation Set Classifier
1. data = collection of N samples
2. thresh = threshold value
3. while(true):
    - Let c_1 be 2-class linear SVM
    - Fit c_1 on data
    - Let c_2 be 1-class non-linear SVM
    - Fit c_2 on(+) side of c_1
    - Add new sample s_i to data
    - If Pr(s_i=(+) | s_i in c_2) > thresh then halt
'''

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

iris = datasets.load_iris()

# Width, height
# Get ride of 3rd class
X = iris.data[50:,:2]
y = iris.target[50:]
y[y == 1] = -1
y[y == 2] = 1

# Fit classifier on inputs and true classes
lin_clf = svm.LinearSVC(max_iter=10000)
lin_clf.fit(X,y)

# Trim inputs to get subset that are (+) from linear classifier
y_sub = lin_clf.predict(X)
X_sub = []
for i, label in enumerate(y_sub):
    if label == 1:
        X_sub.append(X[i])
X_sub = np.array(X_sub)

# sns.distplot(y_sub)
# plt.show()

### TEST
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df)
plt.show()

###

# unique, counts = np.unique(y_pred, return_counts=True)
# print(dict(zip(unique, counts)))
# print(y_pred)

# Fit classifier from subset of inputs that are (+) from previous classifier
one_clf = svm.OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")
one_clf.fit(X_sub, y_sub)

models = (lin_clf, one_clf)

# title for the plots
titles = ('LinearSVC (linear kernel)', 'OneClassSVM (rbf kernel)')

# Set-up 2 grids for plotting.
fig, sub = plt.subplots(2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

# for clf, title, ax in zip(models, titles, sub.flatten()):
#     plot_contours(ax, clf, xx, yy,
#                   cmap=plt.cm.coolwarm, alpha=0.8)
#     ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
#     ax.set_xlim(xx.min(), xx.max())
#     ax.set_ylim(yy.min(), yy.max())
#     ax.set_xlabel('Sepal length')
#     ax.set_ylabel('Sepal width')
#     ax.set_xticks(())
#     ax.set_yticks(())
#     ax.set_title(title)

# plt.show()

'''
Q: how to trim data based on decision boundary?
'''
