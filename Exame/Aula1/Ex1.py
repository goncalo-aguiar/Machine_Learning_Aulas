from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
Y = iris.target

print(Y)

import matplotlib.pyplot as plt

x_min, x_max = X[:,0].min() - 0.5 , X[:,0].max() + 0.5
y_min, y_max = X[:,1].min() - 0.5 , X[:,1].max() + 0.5


fig, ax = plt.subplots()

scatter = ax.scatter(X[:, 0], X[:, 1],c=Y,cmap=plt.cm.Set1,edgecolor= "k")
plt.xlabel ("sepal length")

plt.ylabel("sepal width")


legend1 = ax.legend(*scatter.legend_elements(), loc = "lower right",title = "Class label")
ax.add_artist (legend1)


plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()