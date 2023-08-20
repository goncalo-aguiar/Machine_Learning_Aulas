import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print(
    "explained variance ratio (first two components): %s"
    % str(pca.explained_variance_ratio_)
)

plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("PCA of IRIS dataset")

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r2[y == i, 0], X_r2[y == i, 1], alpha=0.8, color=color, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of IRIS dataset")

# plt.show()


from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


# indices 0-49, 50-99, 100-149 in random order
random0 = np.random.choice(np.arange(0,50),50,replace=False)
random1 = np.random.choice(np.arange(50,100),50,replace=False)
random2 = np.random.choice(np.arange(100,150),50,replace=False)


X = X_r
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


y_true = y01_test
y_pred = clf.predict(X01_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score->",accuracy_score(y_true, y_pred)  )

from sklearn.metrics import f1_score
print("F1 Score->",f1_score(y_true, y_pred)  )

from sklearn.metrics import confusion_matrix
print("Confusion Matrix->")
print(confusion_matrix(y_true, y_pred)  )

from sklearn.metrics import jaccard_score

print("Jaccard Score->",jaccard_score(y_true, y_pred)  )


plt.show()


