

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X = X[Y==0][:,0:2]
Y = Y[Y==0]


random0 = np.random.choice(np.arange(0,50),50, replace=False)

train_inds = random0[:40]
test_inds = random0[40:]

x_min,x_max = X[:,0].min() - 0.5 , X[:,0].max() +0.5
y_min,y_max = X[:,1].min() - 0.5 , X[:,1].max() +0.5


fig, ax = plt.subplots()

scatter = ax.scatter(X[train_inds, 0], X[train_inds, 1])
scatter = ax.scatter(X[test_inds, 0], X[test_inds, 1])
plt.xlabel("sepal length")

plt.ylabel("sepal width")

plt.legend([ "Training examples", "Test examples"])
plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

##-----------------------------------------------------------------------

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X[train_inds,0:1],X[train_inds,1:])
# print(X[train_inds,0:1])
# parameters of the regression line: slope and intercept
print(regr.coef_, regr.intercept_)

# draw the regression line .
line = np.arange(x_min,x_max+1)*regr.coef_[0] + regr.intercept_[0]
plt.plot(np.arange(x_min,x_max+1),line, 'r')

predictions = regr.predict(X[test_inds, 0:1])

from sklearn.metrics import mean_squared_error
print("\nMean Squared Error->",mean_squared_error(predictions, X[test_inds,1:]))

from sklearn.metrics import mean_absolute_error
print("\nMean Absolute Error->",mean_absolute_error(predictions, X[test_inds,1:]))


from sklearn.metrics import explained_variance_score
print("\nExplained Variance Score->",explained_variance_score(predictions, X[test_inds,1:]))


from sklearn.metrics import r2_score
print("\nR2 Score->",r2_score(predictions, X[test_inds,1:]) )

plt.show()