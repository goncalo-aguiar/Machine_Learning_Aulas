from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
Y = iris.target

import numpy as np
import matplotlib.pyplot as plt

# indices 0-49, 50-99, 100-149 in random order

random0 = np. random. choice (np.arange (0,50) ,50, replace=False)
random1 = np.random.choice(np.arange (50, 100), 50, replace=False)
random2 = np. random.choice(np.arange(100,150) ,50, replace=False)

# take 80% (40 samples) of each class to the training set
X0 = X[random0[:40],:]
X1 = X[random1[:40],:]
X2 = X[random2[:40],:]

# take the corresponding labels
Y0 = Y[random0[:40]]
Y1 = Y[random1[:40]]
Y2 = Y[random2[:40]]

# take 20% (10 samples) of each class to the training set
X0_test = X[random0[40:],:]
X1_test = X[random1[40:],:]
X2_test = X[random2[40:],:]

# take the corresponding labels
y0_test = Y[random0[40:]]
y1_test = Y[random1[40:]]
y2_test = Y[random2[40:]]

# compose the training set - just features © and 1 and classes © and 1
X01_train = np.concatenate([X1[:,0:2], X2[:,0:2]])
Y01_train = np.concatenate([Y1,Y2])
X01_test = np.concatenate([X1_test[:,0:2], X2_test[:,0:2]])
Y01_test = np.concatenate([y1_test,y2_test])

from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(random_state=0).fit(X01_train, Y01_train)


b = clf.intercept_[0] 

w1,w2 = clf.coef_.T

c = -b/w2
m = -w1/w2

xmin, xmax = np.min(X01_train,0)[0]-1, np.max(X01_train,0)[0]+1
ymin, ymax = np.min(X01_train,0)[1]-1, np.max(X01_train,0)[1]+1

xd = np.array([xmin,xmax])
yd = m*xd +c

plt.figure()

plt.plot(xd, yd, 'k', lw=1, ls='--')

plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

plt.scatter(*X01_train.T, c=Y01_train, cmap=plt.cm.Set1, edgecolor="k")

plt.scatter(*X01_test.T,c=Y01_test, cmap=plt.cm.Set1, edgecolor="b")

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel(r'sepal width')
plt.xlabel(r'sepal length')

plt.show()