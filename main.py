import numpy as np

np.random.seed(42)

learning_rate = 1e-1
gradient_checking = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cost(y_hat, y):
    return np.mean([_ * _ for _ in (y_hat - y)])

def forward_propagation(X, y, theta, biases):
    theta_1, theta_2, theta_3 = theta[0], theta[1], theta[2]
    bias_1, bias_2, bias_3 = biases[0], biases[1], biases[2]
    a_1 = np.hstack((bias_1, X))
    z_2 = a_1 @ theta_1.T
    a_2 = np.hstack((bias_2, sigmoid(z_2)))
    z_3 = a_2 @ theta_2.T
    a_3 = np.hstack((bias_3, sigmoid(z_3)))
    z_4 = a_3 @ theta_3.T
    a_4 = z_4
    output = a_4
    return cost(output, y)

def approx_assert(a, b, message="", epsilon=0.01):
    assert a - epsilon < b and a + epsilon > b, message

def normalize(d):
    means = np.mean(d, axis=0)
    return d - means

# 0 data
X = np.array([[-3.5,-1.5],[4.5,-4.5],[-2.5,-3.5],[2.5, -2]])
X = normalize(X)
y = np.array([2.25,3.25,0.25,-5.75]).reshape((4,1))
y = normalize(y)
m = X.shape[0]
n = X.shape[1]

# 1 initialize parameters
layer_2 = 5
layer_3 = 4
output = 1

bias_1 = np.ones((m, 1))

# 5 x 2+1
theta_1 = np.random.rand(layer_2, n+1)
bias_2 = np.ones((m, 1))

# 4 x 5+1
theta_2 = np.random.rand(layer_3, layer_2+1)
bias_3 = np.ones((m, 1))

# 1 x 4+1
theta_3 = np.random.rand(output, layer_3+1)

steps = 10
for i in range(steps):
    # 2 forward propagation
    # Layer 1
    # 4 x 2+1
    # a_1 = np.hstack((np.ones((X.shape[0],1)), X))
    a_1 = np.hstack((bias_1, X))
    assert a_1.shape == (4, 3)
    # Layer 2
    # 4 x 3 @ 3 x 5 => 4 x 5
    z_2 = a_1 @ theta_1.T
    assert z_2.shape == (4, 5)
    # 4 x 5+1
    a_2 = np.hstack((bias_2, sigmoid(z_2)))
    assert a_2.shape == (4, 6)
    # Layer 3
    # 4 x 6 @ 6 x 4 => 4 x 4
    z_3 = a_2 @ theta_2.T
    assert z_3.shape == (4, 4)
    # 4 x 4+1
    a_3 = np.hstack((bias_3, sigmoid(z_3)))
    assert a_3.shape == (4, 5)
    # Layer 4
    # 4 x 4+1 @ 5 x 1 => 4 x 1
    z_4 = a_3 @ theta_3.T
    assert z_4.shape == (4, 1)
    # 4 x 1
    a_4 = z_4
    assert a_4.shape == (4, 1)
    output = a_4

    # 3 cost function
    print("cost:", cost(output, y))

    # 4 backward propagation
    # Layer 4
    # 4 x 1
    delta_4 = (output - y)
    assert delta_4.shape == (4, 1)
    # Layer 3
    # 4 x 1 @ 1 x 5 => 4 x 5
    delta_3 = delta_4 @ theta_3 * np.hstack((bias_3, (a_3[:,1:] * (1 - a_3[:,1:]))))
    assert delta_3.shape == (4, 5)
    # 1 x 4 @ 4 x 5 => 1 x 5
    dW_3 = delta_4.T @ a_3
    assert dW_3.shape == (1, 5)
    db_3 = np.array([[1]])#delta_4[:,0].T @ a_3
    # db_3 = np.sum(delta_3, axis=1, keepdims=True)
    # Layer 2
    # 4 x 4 @ 4 x 6 => 4 x 6
    delta_2 = delta_3[:,1:] @ theta_2 * np.hstack((bias_2, (a_2[:,1:] * (1 - a_2[:,1:]))))
    assert delta_2.shape == (4, 6)
    # 4 x 4 @ 4 x 6 => 4 x 6
    dW_2 = delta_3[:,1:].T @ a_2
    assert dW_2.shape == (4, 6)

    db_2 = np.array([[1]])#delta_3[:,0].T @ a_2
    print("delta_3:",delta_3)
    print("delta_2:",delta_2)
    tru = -0.0284127
    epsilon = 1e-3*2
    print("1:",np.mean(delta_2[:,0]), tru - epsilon < np.mean(delta_2[:,0]) and tru + epsilon > np.mean(delta_2[:,0]))
    print("2:",np.mean(delta_2[:,1]), tru - epsilon < np.mean(delta_2[:,1]) and tru + epsilon > np.mean(delta_2[:,1]))
    print("3:",np.mean(delta_2[:,2]), tru - epsilon < np.mean(delta_2[:,2]) and tru + epsilon > np.mean(delta_2[:,2]))
    print("4:",np.mean(delta_2[:,3]), tru - epsilon < np.mean(delta_2[:,3]) and tru + epsilon > np.mean(delta_2[:,3]))
    print("5:",np.mean(delta_2[:,4]), tru - epsilon < np.mean(delta_2[:,4]) and tru + epsilon > np.mean(delta_2[:,4]))
    print("6:",np.mean(delta_2[:,5]), tru - epsilon < np.mean(delta_2[:,5]) and tru + epsilon > np.mean(delta_2[:,5]))

    print("7:",np.mean(delta_2[0,:]), tru - epsilon < np.mean(delta_2[0,:]) and tru + epsilon > np.mean(delta_2[0,:]))
    print("8:",np.mean(delta_2[1,:]), tru - epsilon < np.mean(delta_2[1,:]) and tru + epsilon > np.mean(delta_2[1,:]))
    print("9:",np.mean(delta_2[2,:]), tru - epsilon < np.mean(delta_2[2,:]) and tru + epsilon > np.mean(delta_2[2,:]))
    print("10:",np.mean(delta_2[3,:]), tru - epsilon < np.mean(delta_2[3,:]) and tru + epsilon > np.mean(delta_2[3,:]))

    print("a_2:",a_2)
    # assert db_2.shape == (4, 1)
    # db_2 = np.sum(delta_2, axis=1, keepdims=True)
    
    # Layer 1
    # 5 x 4 @ 4 x 3 => 5 x 3
    dW_1 = delta_2[:,1:].T @ a_1
    assert dW_1.shape == (5, 3)
    db_1 = np.array([[-0.02500924]])
    print("db_1:",db_1)
    # db_1 = np.sum(delta_2, axis=1, keepdims=True)

    def elias(is_theta, params, thetas, biases, real_grads, epsilon):
        for idx, param in enumerate(params):
            use_shape = real_grads[idx].shape
            grad_approx = np.zeros(use_shape)
            for row in range(use_shape[0]):
                for col in range(use_shape[1]):
                    new_param = param.copy()
                    baseline = [0, 0]
                    for i, diff in enumerate([epsilon, -epsilon * 2]):
                        new_param[row,col] += diff
                        tmp_params = params.copy()
                        tmp_params[idx] = new_param
                        baseline[i] = forward_propagation(X, y, tmp_params if is_theta else thetas, tmp_params if not is_theta else biases)
                    if not is_theta:
                        print("row:",row)
                        print("col:",col)
                        print("grad_approx:",grad_approx, grad_approx.shape)
                        print("baseline:",baseline)
                    grad_approx[row,col] = (baseline[0] - baseline[1]) / (2 * epsilon)
            real = (learning_rate * real_grads[idx]) / (2 * learning_rate)
            print(f"{'theta' if is_theta else 'bias'}_{idx+1}")
            print("approx:",grad_approx)
            print('real:',real)
            print()
            numerator = np.linalg.norm(grad_approx - real)
            denominator = np.linalg.norm(grad_approx) + np.linalg.norm(real)
            res = numerator / denominator
            approx_assert(0, res, f"{'theta' if is_theta else 'bias'}_{idx+1}", epsilon)



    # 4.1 gradient checking
    if gradient_checking:
        epsilon = 1e-3
        elias(True, [theta_1, theta_2, theta_3], [theta_1, theta_2, theta_3], [bias_1, bias_2, bias_3], [dW_1, dW_2, dW_3], epsilon)
        elias(False, [bias_1, bias_2, bias_3], [theta_1, theta_2, theta_3], [bias_1, bias_2, bias_3], [db_1, db_2, db_3], epsilon)

    # 5 update parameters
    theta_3 -= learning_rate * dW_3
    theta_2 -= learning_rate * dW_2
    theta_1 -= learning_rate * dW_1

    # Analytics
    pass
