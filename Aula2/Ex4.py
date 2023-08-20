# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:07:53 2022

@author: Kuba
"""

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data
y = iris.target


x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("sepal length")
plt.ylabel("sepal width")

legend1 = ax.legend(*scatter.legend_elements(),loc="lower right", title="Class")
ax.add_artist(legend1)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)


# indices 0-49, 50-99, 100-149 in random order
random0 = np.random.choice(np.arange(0,50),50,replace=False)
random1 = np.random.choice(np.arange(50,100),50,replace=False)
random2 = np.random.choice(np.arange(100,150),50,replace=False)

# take 80% (40 samples) of each class to the training set

X0 = X[random0[:40],:]
X1 = X[random1[:40],:]
X2 = X[random2[:40],:]

# take the corresponding labels
y0 = y[random0[:40]]
y1 = y[random1[:40]]
y2 = y[random2[:40]]

# take 20% (10 samples) of each class to the test set
X0_test = X[random0[40:],:]
X1_test = X[random1[40:],:]
X2_test = X[random2[40:],:]

# take the corresponding labels
y0_test = y[random0[40:]]
y1_test = y[random1[40:]]
y2_test = y[random2[40:]]

# compose the training set and the test set - just features 0 and 1 and classes 0 and 1
X01_train = np.concatenate([X0[:,0:2], X1[:,0:2]])
y01_train = np.concatenate([y0, y1])
X01_test = np.concatenate([X0_test[:,0:2], X1_test[:,0:2]])
y01_test = np.concatenate([y0_test, y1_test])

# linear binary classification with a logistic regression model
clf = LogisticRegression(random_state=0).fit(X01_train, y01_train)

# read the logistic regression model parameters
b = clf.intercept_[0]
w1, w2 = clf.coef_.T


# calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

# plot the data and the classification with the decision boundary.
xmin, xmax = np.min(X01_train,0)[0]-1, np.max(X01_train,0)[0]+1
ymin, ymax = np.min(X01_train,0)[1]-1, np.max(X01_train,0)[1]+1

xd = np.array([xmin, xmax])
yd = m*xd + c

plt.figure()
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

plt.scatter(*X01_train.T, c=y01_train, cmap=plt.cm.Set1, edgecolor="k")
plt.scatter(*X01_test.T, c=y01_test, cmap=plt.cm.Set1, edgecolor="b")

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel(r'sepal width')
plt.xlabel(r'sepal length')




probs = clf.predict_proba(X01_test)

y_pred = clf.predict(X01_test)

for i in range(len(y01_test)):
    print(f"Sample {i}: probabilities {probs[i]}, predicted class {y_pred[i]}, actual class {y01_test[i]}")

plt.show()