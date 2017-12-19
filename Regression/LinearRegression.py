import numpy as np
from numpy import random
from numpy import matlib
from numpy import linalg

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
import tensorflow as tf

# Model parameters
w = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x_tf = tf.placeholder(tf.float32)
linear_model = w*x_tf + b
y_tf = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y_tf)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
# x_train = [1, 2, 3, 4]
# y_train = [0, -1, -2, -3]
x_train = x.T
y_train = y.T
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x_tf: x_train, y_tf: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([w, b, loss], {x_tf: x_train, y_tf: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

# W_tf = np.concatenate([curr_W,curr_b],axis=0)
W_tf = [curr_W,curr_b]

print(W_tf)






# Fit Linear Regression with TensorFlow
# W_tf = tf.Variable(np.matlib.ones([2,1],'Float64'))
# x_tf = tf.placeholder(tf.float64)
# y_tf = tf.placeholder(tf.float64)
# y_hat = x_tf*W_tf
#
# loss = tf.reduce_sum(tf.square(y_hat-y_tf))
#
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
#
# sess = tf.Session()
# sess.run(train,{x_tf:x_homo, y_hat:y})
#
# # Train Model
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# for i in range(1,1000):
#     sess.run(train,{x_tf:x_homo, y_hat:y})



# Visualize Points Scatter
[fig,ax] = plt.subplots()
ax.plot(x,y,linestyle='',marker='o',color='green')

## Visualize the fitted line
x_range = np.matrix([[0, 1],[1,1]])
x_range = x_range.T
y_range = np.matmul(x_range,W)

y_range_tf = np.matmul(x_range,np.matrix(W_tf))

ax.plot([0,1],y_range,linestyle='-')
ax.plot([0,1],y_range_tf,linestyle='--',color='red')

plt.show()