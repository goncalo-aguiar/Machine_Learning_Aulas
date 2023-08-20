# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 12:06:04 2022

@author: Kuba
"""

# Task: modify the example below
# to predict sepal width based on two other features – 
# sepal length (0) and petal length (2). Fill in the program in places marked
# with !!!. In your report, note the values of the learned line parameters and 
# show the plot

# THE CODE LACKS:
    # an import of the proper submodule from sklearn that containes linear models
    # a definition of the X array containing examples of class 0 (setosa) and the first
    # three features only
    # a definition of the linear regression object, to be named multiregr
    # a line of code to train the model

# importing necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
# !!!

iris = datasets.load_iris()

X = iris.data
Y = iris.target

# take only examples from class 0 (iris setosa) and first 3 features
# !!!
X = X[Y == 0][:,0:3]
Y = Y[Y==0]
# random ordering of samples
random0 = np.random.choice(np.arange(0,50),50,replace=False)

# test and training set example indices
train_inds = random0[:40]
test_inds = random0[40:]

# create the regression model and train it
# !!!
# !!!
from sklearn import linear_model
multiregr = linear_model.LinearRegression()


multiregr.fit(X[train_inds,:][:,[0,2]],X[train_inds,1])

# check the learned parameters
print(multiregr.coef_, multiregr.intercept_) 

# feature value ranges (for the plot)
x_min, x_max = X[train_inds, 0].min() - 0.5, X[train_inds, 0].max() + 0.5
x_min2, x_max2 = X[train_inds, 2].min() - 0.5, X[train_inds, 2].max() + 0.5

# create a grid of points (x,y) according to the data value ranges
x, y = np.meshgrid(np.arange(x_min,x_max,0.5), np.arange(x_min2,x_max2,0.5))
surface_points = np.stack([np.ravel(x), np.ravel(y)],1)

# predict the dependent variable value
z = multiregr.predict(surface_points).reshape(x.shape)

# 3D graph
fig = plt.figure(figsize=plt.figaspect(1)*2)
ax = plt.axes(projection='3d')
ax.scatter(X[train_inds, 0],X[train_inds, 2],  X[train_inds, 1],label = "Training examples")
ax.scatter(X[test_inds, 0],X[test_inds, 2],  X[test_inds, 1], label = "Test examples")
ax.plot_surface(x, y, z,  color = 'r', alpha = 0.4)
ax.legend()
ax.set_xlabel("Sepal length")
ax.set_ylabel("Petal length")
ax.set_zlabel("Sepal width (dependent variable)")
plt.show()