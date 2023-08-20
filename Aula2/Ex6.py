# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 17:46:12 2022

@author: Kuba

This example uses code from the University of Warsaw educational materials:
https://colab.research.google.com/drive/1QTOK6B8jPrP3J7NW2I93q3WZJ1CAWN_H

"""

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
import numpy as np

# generating a 'circularly distributed' dataset and target class labels
X, y = make_circles(500, factor=.1, noise=.1)

# scatter plot of the training data
ax = plt.gca()
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.Set1, edgecolor="k")
ax.set_xlabel('Some feature 1')
ax.set_ylabel('Some feature 2')

# creating an instance of the support vector machine classifier
# with a radial basis function type of kernel (nonlinear)
svm = SVC(kernel='rbf', C=1E10)

# training
svm.fit(X, y)

# grid of points for drawing the nonlinear decision boundary
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# drawing the decision bounary (black line) and margins (blue line) using the .decision_function() method
Z = svm.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors=['b', 'k', 'b'], levels=[-1, 0, 1], alpha=1,
           linestyles=['--', '-', '--'])

# mark support vectors (black crosses)
support_vector1 = svm.support_vectors_[:, 0]
support_vector2 = svm.support_vectors_[:, 1]
ax.scatter(support_vector1, support_vector2, s=50,
           linewidth=1, color='k', marker = 'x')

plt.show()