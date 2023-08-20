



import numpy as np
import matplotlib.pyplot as plt

# load data from .npy file
X = np.load('data.npy')
Y = np.load('targets.npy')

# do something with the data
print(X)
print(Y)

x_min, x_max = X[:,3].min() -50, X[:,3].max() +50
y_min, y_max = X[:,29].min() -0.05 , X[:,29].max() +0.05


fig, ax = plt.subplots()

scatter = ax.scatter(X[:, 3], X[:, 29],c=Y,cmap=plt.cm.Set1,edgecolor= "k")
plt.xlabel ("mean area")

plt.ylabel("worst fractal dimension")
plt.title("Breast Cancer Dataset")

legend1 = ax.legend(*scatter.legend_elements(), loc = "lower right",title = "Class label")
ax.add_artist (legend1)


plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_r, Y, test_size=0.2, random_state=101)


from sklearn.svm import SVC

# create an SVM classifier with RBF kernel and regularization constant C=0.5
svm = SVC(kernel='linear', C=0.5, random_state=101)

# train the classifier on the training data
svm.fit(X_train, Y_train)

# make predictions on the test data
Y_pred = svm.predict(X_test)


from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()

print("There were", tn, "true negatives,", tp, "true positives,", fn, "false negatives and", fp, "false positives in the test set results.")


b = svm.intercept_[0]
w1, w2 = svm.coef_.T

# calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

# plot the data and the classification with the decision boundary.
xmin, xmax = np.min(X_train,0)[0]-50, np.max(X_train,0)[0]+50
ymin, ymax = np.min(X_train,0)[1]-50, np.max(X_train,0)[1]+50

xd = np.array([xmin, xmax])
yd = m*xd + c

plt.figure()
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

plt.scatter(*X_train.T, c=Y_train, cmap=plt.cm.Set1, edgecolor="k")
plt.scatter(*X_test.T, c=np.array(Y_test, dtype=float), cmap=plt.cm.Set1, edgecolor="b")

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel(r'2nd principal component')
plt.xlabel(r'1st principal component')
plt.title("Breast cancer classification")

plt.show()




