import numpy as np

learning_rate = 0.01
gradient_checking = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 0 data
# TODO: normalize
X = np.array([[3,5],[10,2],[4,3],[9, -2]]).T
y = np.array([34,33,26,10]).reshape((1, 4))
n = X.shape[0]
m = X.shape[1]

# 1 initialize parameters
layer_2 = 5
layer_3 = 4
output = 1

bias_1 = np.random.randn(1, m)

# 5 x 3
theta_1 = np.random.randn(layer_2, n+1)
bias_2 = np.random.randn(1, m)

# 4 x 6
theta_2 = np.random.randn(layer_3, layer_2+1)
bias_3 = np.random.randn(1, m)

# 1 x 5
theta_3 = np.random.randn(output, layer_3+1)

steps = 10
for i in range(steps):
    # 2 forward propagation
    # Shape: (n+1) x m => 3 x 4
    a_1 = np.concatenate((X, bias_1))
    # Shape: (layer_2+1) x m => 6 x 4
    a_2 = np.concatenate((sigmoid(theta_1 @ a_1), bias_2))
    # Shape: (layer_3+1) x m => 5 x 4
    a_3 = np.concatenate((sigmoid(theta_2 @ a_2), bias_3))
    # Shape: output x m => 1 x 4
    output = theta_3 @ a_3

    # 3 cost function

    # 4 backward propagation
    # output x m
    # 1 x 4
    delta_4 = output - y
    # (layer_3+1) x output @ output x m * (layer_3+1) x m
    # 5 x 1 @ 1 x 4 * 5 x 4 => 5 x 4
    delta_3 = theta_3.T @ delta_4 * (a_3 * (1 - a_3))
    # (layer_2+1) x layer_3 @ (layer_3+1) x m * (layer_2+1) x m
    # 6 x 4 @ 5 x 4 * 6 x 4 => 6 x 4
    delta_2 = theta_2.T @ delta_3 * (a_2 * (1 - a_2))
    # (n+1) x layer_2 @ (layer_2+1) x m * (n+1) x m
    # 3 x 5 @ 6 x 4 * 3 x 4 => 5 x 3
    delta_1 = theta_1.T @ delta_2 * (a_1 * (1 - a_1))

    # 4.1 gradient checking

    # 5 update parameters
    theta_2 += learning_rate * delta_2 @ a_1
    theta_3 += learning_rate * delta_3 @ a_2
    theta_4 += learning_rate * delta_4 @ a_3

    # Analytics
    pass
