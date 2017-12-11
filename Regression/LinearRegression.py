import numpy as np
from numpy import random
from numpy import matlib
from numpy import linalg

import tensorflow as tf

import matplotlib.pyplot as plt


NUM_POINTS = 10

x = random.random([NUM_POINTS,1])
x = np.matrix(x,'Float64')


print(x)

# synthesize y
a = 3;
b = 1*random.randn();
error = 0.1*random.randn(NUM_POINTS,1)
y = a*x+b+error

print(y)

# Fit Linear Regression with Numpy
x_homo = np.concatenate([x,matlib.ones([NUM_POINTS,1],'Float64')],axis=1)


print(x_homo)

W = linalg.inv(x_homo.T*x_homo)*x_homo.T*y

print(W)

# Fit Linear Regression with TensorFlow


# Visualize Points Scatter
[fig,ax] = plt.subplots()
ax.plot(x,y,linestyle='',marker='o',color='green')

## Visualize the fitted line
x_range = np.matrix([[0, 1],[1,1]])
x_range = x_range.T
y_range = x_range*W

ax.plot([0,1],y_range,linestyle='-')

plt.show()