import numpy as np

np.random.seed(42)
learning_rate = 0.01
gradient_checking = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def J(X, y, theta_1, bias_1, theta_2, bias_2, theta_3, bias_3):
    a_1 = X + bias_1
    a_2 = sigmoid(theta_1 @ a_1 + bias_2)
    a_3 = sigmoid(theta_2 @ a_2 + bias_3)
    output = theta_3 @ a_3
    return output - y

def approx_assert(a, b, message="", epsilon=0.01):
    print("a.shape:",a.shape)
    print("b.shape:",b.shape)
    assert (a - epsilon < b).all() and (a + epsilon > b).all(), message

# 0 data
# TODO: normalize
X = np.array([[-3.5,-1.5],[3.5,-4.5],[-2.5,-3.5],[2.5, -2]]).T
y = np.array([8.25,9.25,0.25,-15.75]).reshape((1, 4))
n = X.shape[0]
m = X.shape[1]

# 1 initialize parameters
layer_2 = 5
layer_3 = 4
output = 1

bias_1 = np.random.rand(n, 1)

# 5 x 2
theta_1 = np.random.rand(layer_2, n)
bias_2 = np.random.rand(layer_2, 1)

# 4 x 5
theta_2 = np.random.rand(layer_3, layer_2)
bias_3 = np.random.rand(layer_3, 1)

# 1 x 4
theta_3 = np.random.rand(output, layer_3)

steps = 10
for i in range(steps):
    # 2 forward propagation
    # Layer 1
    # Shape: (n+1) x m => 4 x 3
    a_1 = X + bias_1
    # Layer 2
    # 5 x 3 @ 3 x 4 => 5 x 4
    z_2 = theta_1 @ a_1 + bias_2
    # Shape: layer_2 x m => 5 x 4
    a_2 = sigmoid(z_2)
    # Layer 3
    # 4 x 6 @ 5 x 4 => 4 x 4
    z_3 = theta_2 @ a_2 + bias_3
    # Shape: layer_3 x m => 4 x 4
    a_3 = sigmoid(z_3)
    # Layer 4
    # Shape: output x m => 1 x 4
    a_4 = theta_3 @ a_3
    output = a_4

    # 3 cost function

    # 4 backward propagation
    # Layer 4
    # 1 x 4
    delta_4 = (a_4 - y) * (a_4 * (1 - a_4))
    print("output error:",delta_4)
    # Layer 3
    dZ_3 = delta_4
    print("dZ_3:",dZ_3)
    # 1 x 4 @ 4 x 1 => 1 x 4
    dW_3 = dZ_3 @ a_3.T * (a_4 * (1 - a_4))
    print("dW_3:", dW_3)
    db_3 = np.sum(dZ_3, axis=1, keepdims=True)
    # Layer 2
    # 4 x 1 @ 1 x 4 => 4 x 4
    dZ_2 = theta_3.T @ dZ_3 * (a_3 * (1 - a_3))
    print("dZ_2:",dZ_2)
    # 4 x 4 @ 4 x 5 => 4 x 5
    dW_2 = dZ_2 @ a_2.T
    print("dW_2:", dW_2)
    db_2 = np.sum(dZ_2, axis=1, keepdims=True)
    # Layer 1
    # 5 x 4 @ 4 x 4 => 5 x 4
    dZ_1 = theta_2.T @ dZ_2 * (a_2 * (1 - a_2))
    print("dZ_1:",dZ_1)
    # 5 x 4 @ 4 x 2 => 5 x 2
    dW_1 = dZ_1 @ a_1.T
    print("dW_1:", dW_1)
    db_1 = np.sum(dZ_1, axis=1, keepdims=True)

    # 4.1 gradient checking
    if gradient_checking:
        epsilon = 1e-3
        baseline = J(X, y, theta_1, bias_1, theta_2, bias_2, theta_3, bias_3)
        assert (baseline == delta_4).all()
        new_theta = theta_1.copy()
        new_theta[0,0] += epsilon
        print("ptheta:",new_theta)
        baseline_plus = J(X, y, new_theta, bias_1, theta_2, bias_2, theta_3, bias_3)
        new_theta[0,0] -= epsilon * 2
        print("mtheta:",new_theta)
        baseline_minus = J(X, y, new_theta, bias_1, theta_2, bias_2, theta_3, bias_3)
        print("baseline_p:",baseline_plus)
        print("baseline_m:",baseline_minus)
        print("baseline_plus - baseline_minus:",baseline_plus - baseline_minus)
        grad_approx = (baseline_plus - baseline_minus) / (2 * epsilon)
        print("grad_approx:",grad_approx)
        print("dW_3:",dW_3 * learning_rate / (2 * epsilon))
        print("db_3:",db_3)
        gan = np.linalg.norm(grad_approx)
        dwn = np.linalg.norm(dW_3)
        print("gan:",gan)
        print("dwn:",dwn)
        upr = np.linalg.norm(grad_approx - dW_3)
        print("upr:",upr)
        res = upr / (gan + dwn)
        print('res:',res)
        approx_assert(grad_approx, dW_3, "baseline", epsilon)

        for r in range(len(theta_1)):
            for c in range(len(theta_1[r])):
                new_theta = theta_1.copy()
                new_theta[r,c] = theta_1[r,c] + epsilon
                pt = J(X, y, new_theta, bias_1, theta_2, bias_2, theta_3, bias_3)
                print("pt:",pt)
                new_theta[r,c] = theta_1[r,c] - epsilon * 2
                mt = J(X, y, new_theta, bias_1, theta_2, bias_2, theta_3, bias_3)
                print("mt:",mt)
                grad_approx = (pt - mt) / (2 * epsilon)
                print("grad_approx:",grad_approx)
                print("dW_3:",dW_3)
                approx_assert(grad_approx, dW_3, f'theta 1 => r: {r}, c: {c}', epsilon)
        # for r in range(len(bias_1)):
        #     for c in range(len(bias_1[r])):
        #         new_bias = bias_1.copy()
        #         new_bias[r,c] = bias_1[r,c] + epsilon
        #         pt = J(X, y, theta_1, new_bias, theta_2, bias_2, theta_3, bias_3)
        #         new_bias[r,c] = bias_1[r,c] - epsilon
        #         mt = J(X, y, theta_1, new_bias, theta_2, bias_2, theta_3, bias_3)
        #         grad_approx = (pt - mt) / (2 * epsilon)
        #         approx_assert(grad_approx, db_1, f'bias 1 => r: {r}, c: {c}')
        # #
        # for r in range(len(theta_2)):
        #     for c in range(len(theta_2[r])):
        #         new_theta = theta_2.copy()
        #         new_theta[r,c] = theta_2[r,c] + epsilon
        #         pt = J(X, y, theta_1, bias_1, new_theta, bias_2, theta_3, bias_3)
        #         new_theta[r,c] = theta_2[r,c] - epsilon
        #         mt = J(X, y, theta_1, bias_1, new_theta, bias_2, theta_3, bias_3)
        #         grad_approx = (pt - mt) / (2 * epsilon)
        #         approx_assert(grad_approx, dW_2, f'theta 2 => r: {r}, c: {c}')
        # for r in range(len(bias_2)):
        #     for c in range(len(bias_2[r])):
        #         new_bias = bias_2.copy()
        #         new_bias[r,c] = bias_2[r,c] + epsilon
        #         pt = J(X, y, theta_1, bias_1, theta_2, new_bias, theta_3, bias_3)
        #         new_bias[r,c] = bias_2[r,c] - epsilon
        #         mt = J(X, y, theta_1, bias_1, theta_2, new_bias, theta_3, bias_3)
        #         grad_approx = (pt - mt) / (2 * epsilon)
        #         approx_assert(grad_approx, db_2, f'bias 2 => r: {r}, c: {c}')
        # #
        # for r in range(len(theta_3)):
        #     for c in range(len(theta_3[r])):
        #         new_theta = theta_3.copy()
        #         new_theta[r,c] = theta_3[r,c] + epsilon
        #         pt = J(X, y, theta_1, bias_1, theta_2, bias_2, new_theta, bias_3)
        #         new_theta[r,c] = theta_3[r,c] - epsilon
        #         mt = J(X, y, theta_1, bias_1, theta_2, bias_2, new_theta, bias_3)
        #         grad_approx = (pt - mt) / (2 * epsilon)
        #         approx_assert(grad_approx, dW_3, f'theta 3 => r: {r}, c: {c}')
        # for r in range(len(bias_3)):
        #     for c in range(len(bias_3[r])):
        #         new_bias = bias_3.copy()
        #         new_bias[r,c] = bias_3[r,c] + epsilon
        #         pt = J(X, y, theta_1, bias_1, theta_2, bias_2, theta_3, new_bias)
        #         new_bias[r,c] = bias_3[r,c] - epsilon
        #         mt = J(X, y, theta_1, bias_1, theta_2, bias_2, theta_3, new_bias)
        #         grad_approx = (pt - mt) / (2 * epsilon)
        #         approx_assert(grad_approx, db_3, f'bias 3 => r: {r}, c: {c}')

    # 5 update parameters
    theta_3 -= learning_rate * dW_3
    bias_3 -= learning_rate * db_3
    theta_2 -= learning_rate * dW_2
    bias_2 -= learning_rate * db_2
    theta_1 -= learning_rate * dW_1
    bias_1 -= learning_rate * db_1

    # Analytics
    pass
