import numpy as np

np.random.seed(42)

learning_rate = 1e-1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def approx_assert(a, b, message="", epsilon=0.01):
    assert a - epsilon < b and a + epsilon > b, message

def normalize(d):
    means = np.mean(d, axis=0)
    return d - means

# 0 data
X = np.array([[-3.5,-1.5],[4.5,-4.5],[-2.5,-3.5]]).reshape((2, 3))
X = normalize(X)
y = np.array([2.25,3.25,0.25]).reshape((1, 3))
y = normalize(y)
n = X.shape[0]
m = X.shape[1]

# 1 initialize parameters
layer_1 = 5
layer_2 = 4
output = 1

# 5 x 2
weight_1 = np.random.rand(layer_1, n)
bias_1 = np.ones((layer_1, 1))

# 4 x 5
weight_2 = np.random.rand(layer_2, layer_1)
bias_2 = np.ones((layer_2, 1))

# 1 x 4
weight_3 = np.random.rand(output, layer_2)
bias_3 = np.ones((output, 1))

weights = [weight_1, weight_2, weight_3]
biases = [bias_1, bias_2, bias_3]

steps = 10
for i in range(steps):
    # 2 forward propagation
    def forward_propagation(X, y, weights, biases):
        [weight_1, weight_2, weight_3] = weights
        [bias_1, bias_2, bias_3] = biases
        # Layer 1
        # 5 x 2 @ 2 x 3 => 5 x 3
        z_1 = weight_1 @ X + bias_1
        a_1 = sigmoid(z_1)
        assert a_1.shape == (5, 3)
        # Layer 2
        # 4 x 5 @ 5 x 3 => 4 x 3
        z_2 = weight_2 @ a_1 + bias_2
        a_2 = sigmoid(z_2)
        assert a_2.shape == (4, 3)
        # Layer 3
        # 1 x 4 @ 4 x 3 => 1 x 3
        a_3 = weight_3 @ a_2 + bias_3
        assert a_3.shape == (1, 3)
        output = a_3
        return output, [z_1, z_2], [a_1, a_2, a_3]

    output, [z_1, z_2], [a_1, a_2, a_3] = forward_propagation(X, y, weights, biases)

    # 3 cost function
    def cost(y_hat, y):
        return np.mean([_ * _ for _ in (y_hat - y)])
    print("cost:", cost(output, y))

    # 4 backward propagation
    # Layer 3
    # 1 x 3
    dZ_3 = output - y # add activation derivative if used
    assert dZ_3.shape == (1, 3)
    # 1 x 3 @ 3 x 4 => 1 x 4
    dW_3 = dZ_3 @ a_2.T / m
    assert dW_3.shape == weight_3.shape
    # 1 x 3 => 1 x 1
    db_3 = np.sum(dZ_3, axis=1, keepdims=True) / m
    assert db_3.shape == bias_3.shape

    # Layer 2
    # 4 x 1 @ 1 x 3 => 4 x 3
    dZ_2 = weight_3.T @ dZ_3 * sigmoid(z_2) * (1 - sigmoid(z_2))
    assert dZ_2.shape == (4, 3)
    # 4 x 3 @ 3 x 5 => 1 x 5
    dW_2 = dZ_2 @ a_1.T / m
    assert dW_2.shape == weight_2.shape
    # 4 x 3 => 4 x 1
    db_2 = np.sum(dZ_2, axis=1, keepdims=True) / m
    assert db_2.shape == bias_2.shape

    # Layer 1
    # 5 x 4 @ 4 x 3 => 5 x 3
    dZ_1 = weight_2.T @ dZ_2 * sigmoid(z_1) * (1 - sigmoid(z_1))
    assert dZ_1.shape == (5, 3)
    # 5 x 3 @ 3 x 2 => 5 x 2
    dW_1 = dZ_1 @ X.T / m
    assert dW_1.shape == weight_1.shape
    # 5 x 3 => 5 x 1
    db_1 = np.sum(dZ_1, axis=1, keepdims=True) / m
    assert db_1.shape == bias_1.shape

    # 4.1 gradient checking
    def gradient_checking(is_weight, weights, biases, real_grads, epsilon):
        params = weights.copy() if is_weight else biases.copy()
        for idx, param in enumerate(params):
            shape = real_grads[idx].shape
            grad_approx = np.zeros(shape)
            for row in range(shape[0]):
                for col in range(shape[1]):
                    new_param = param.copy()
                    baseline = []
                    for diff in [epsilon, -epsilon * 2]:
                        new_param[row,col] += diff
                        tmp_params = params.copy()
                        tmp_params[idx] = new_param
                        output, _, _ = forward_propagation(X, y, tmp_params if is_weight else weights, tmp_params if not is_weight else biases)
                        baseline.append(cost(output, y))
                    grad_approx[row,col] = (baseline[0] - baseline[1]) / (2 * epsilon)
            real = real_grads[idx] * 2
            numerator = np.linalg.norm(grad_approx - real)
            denominator = np.linalg.norm(grad_approx) + np.linalg.norm(real)
            res = numerator / denominator
            approx_assert(0, res, f"{'weight' if is_weight else 'bias'}_{idx+1}", epsilon)

    epsilon = 1e-3
    gradient_checking(True, weights.copy(), biases.copy(), [dW_1, dW_2, dW_3], epsilon)
    gradient_checking(False, weights.copy(), biases.copy(), [db_1, db_2, db_3], epsilon)

    # 5 update parameters
    weight_3 -= learning_rate * dW_3
    weight_2 -= learning_rate * dW_2
    weight_1 -= learning_rate * dW_1
