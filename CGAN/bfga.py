# coding=utf-8
import numpy as np
import os


def X3(a, b, c):
    a = np.dot(np.dot(a, b), c)
    return a


def X2(a, b):
    a = np.dot(a, b)
    return a


def get_data(obj_path_name):
    pro_path = os.path.abspath('.')
    data_path = str(pro_path + obj_path_name)
    print data_path
    data = np.loadtxt(data_path)
    x = np.asarray(np.column_stack((np.ones((data.shape[0], 1)), data[:, 0:-1])), dtype='double')
    y = data[:, -1]
    y = np.reshape(y, (data.shape[0], 1))
    print x.shape, y.shape
    return x, y


def inverse_hessian_mat(x):
    # 计算hessian矩阵，并求其逆矩阵
    x_hessian = np.dot(x.T, x)
    inverse_hessian = np.linalg.inv(x_hessian)
    return inverse_hessian


def get_e(x, theta, y):
    y_pre = np.dot(x, theta)
    e = y_pre - y

    return e


def compute_grad(x, e, stand_flag=0):
    # batch法必须做梯度归一化
    print e.T.shape, x.shape
    grad = np.dot(e.T, x)
    # grad = np.dot(e.T, x)/x.shape[0]
    if stand_flag == 1:
        grad = grad/(np.dot(grad, grad.T)**0.5)
    return np.reshape(grad, (x.shape[1], 1))


def get_cost(e):
    # print e.shape
    # 计算当前theta组合点下的各个样本的预测值 y_pre
    cost = np.dot(e.T, e) / 2
    # cost = np.dot(e.T, e) / (2*e.shape[0])
    return cost


def get_cost_l12(e, theta, m, l=1, lmd=10e10):
    # print e.shape
    if l == 1:
        cost = (np.dot(e.T, e) + lmd*sum(abs(theta))) / (2*m)
    elif l == 2:
        cost = (np.dot(e.T, e) + lmd*np.dot(theta.T*theta)) / (2*m)
    else:
        cost = (np.dot(e.T, e)) / (2*m)
    return cost


def update_batch_theta(theta, grad, alpha):
    theta = theta - alpha*grad
    return theta


def update_batch_theta_l12(theta, grad, alpha, m, l=1, lmd=10e10):
    if l == 1:
        theta = theta - alpha * (grad + (lmd/m)*theta)
    elif l == 2:
        theta = theta - alpha * (grad + (lmd/m))
    else:
        theta = theta - alpha * grad
    return theta


def iter_batch(x, theta, y, out_n, out_e_reduce_rate, alpha):
    step = 1
    while step < out_n:
        """计算初始的损失值"""
        if step == 1:
            e = get_e(x, theta, y)
            cost_0 = get_cost(e)
        """计算当前theta组合下的grad值"""
        grad = compute_grad(x, e, stand_flag=1)
        """依据grad更新theta"""
        theta = update_batch_theta(theta, grad, alpha)
        """计算新的损失值"""
        e = get_e(x, theta, y)
        cost_1 = get_cost(e)

        e_reduce_rate = abs(cost_1 - cost_0)/cost_0
        print 'Step: %-6s, cost: %s' % (step, cost_1[0, 0])
        if e_reduce_rate < out_e_reduce_rate:
            flag = 'Finish!\n'
            print flag
            break
        else:
            cost_0 = cost_1
            step += 1

    return theta


def iter_random_batch(x, theta, y, out_n, out_e_reduce_rate, alpha):

    step = 0
    while step < out_n:
        step += 1
        for i in range(x.shape[0]):
            x_i = np.reshape(x[i, ], (1, x.shape[1]))
            y_i = np.reshape(y[i, ], (1, 1))
            """计算初始的损失值"""
            e_0 = get_e(x, theta, y)
            cost_0 = get_cost(e_0)

            """用一个样本，计算当前theta组合下的grad值"""
            e_i = get_e(x_i, theta, y_i)
            grad = compute_grad(x_i, e_i, stand_flag=1)
            """依据grad更新theta"""
            theta = update_batch_theta(theta, grad, alpha)
            """计算新的损失值"""
            e_1 = get_e(x, theta, y)
            cost_1 = get_cost(e_1)

            e_reduce_rate = abs(cost_1 - cost_0)/cost_0
            if e_reduce_rate < out_e_reduce_rate:
                flag = 'Finish!\n'
                print flag
                step = out_n + 1
                break
        print 'Step: %-6s, cost: %s' % (step, cost_1[0,0])

    return theta


def iter_mini_batch(x, theta, y, out_n, out_e_reduce_rate, alpha, batch):
    batch_n = x.shape[0]//batch
    batch_left_n = x.shape[0] % batch

    step = 0
    while step < out_n:
        step += 1
        for i in range(batch_n+1):
            """计算初始的损失值"""
            e_0 = get_e(x, theta, y)
            cost_0 = get_cost(e_0)

            """选取更新梯度用的batch个样本"""
            if i <= batch_n-1:
                start = i*batch
                x_i = np.reshape(x[start:start+batch, ], (batch, x.shape[1]))
                y_i = np.reshape(y[start:start+batch, ], (batch, 1))
            else:
                if batch_left_n != 0:
                    x_i = np.reshape(x[-1*batch_left_n:, ], (batch_left_n, x.shape[1]))
                    y_i = np.reshape(y[-1*batch_left_n:, ], (batch_left_n, 1))

            """用batch个样本，计算当前theta组合下的grad值"""
            e_i = get_e(x_i, theta, y_i)
            grad = compute_grad(x_i, e_i, stand_flag=1)
            """依据grad更新theta"""
            theta = update_batch_theta(theta, grad, alpha)
            """计算新的损失值"""
            e_1 = get_e(x, theta, y)
            cost_1 = get_cost(e_1)

            e_reduce_rate = abs(cost_1 - cost_0)/cost_0
            if e_reduce_rate < out_e_reduce_rate:
                flag = 'Finish!\n'
                print flag
                step = out_n
                break
        print 'Step: %-6s, cost: %s' % (step, cost_1[0, 0])

    return theta


def update_newton_theta(x_hessian_inv, theta, grad, alpha):
    theta = theta - alpha*np.dot(x_hessian_inv, grad)
    # print 'New Theta --> ', theta1
    return theta


def iter_newton(x, theta, y, out_n, out_e_reduce_rate, alpha):
    e = get_e(x, theta, y)
    cost_0 = get_cost(e)
    x_hessian_inv = inverse_hessian_mat(x)

    step = 1
    while step < out_n:
        grad = compute_grad(x, e)
        theta = update_newton_theta(x_hessian_inv, theta, grad, alpha)
        e = get_e(x, theta, y)
        cost_1 = get_cost(e)

        print 'Step: %-6s, cost: %s' % (step, cost_1[0,0])
        e_reduce_rate = abs(cost_1 - cost_0)/cost_0
        if e_reduce_rate < out_e_reduce_rate:
            flag = 'Finish!\n'
            print flag
            break
        else:
            cost_0 = cost_1
            step += 1

    return theta


def iter_batch_linesearch(x, theta_0, y, out_n, out_e_reduce_rate, alpha):
    e_0 = get_e(x, theta_0, y)
    cost_0 = get_cost(e_0)

    step = 1
    while step < out_n:
        grad = compute_grad(x, e_0, stand_flag=1)
        theta_1, cost_1, e_1, count = line_search(x, y, alpha, theta_0, grad)
        e_reduce_rate = abs(cost_1 - cost_0)/cost_0
        print 'Step: %-6s, count: %-4s, cost: %s' % (step, count, cost_1[0, 0])
        if e_reduce_rate < out_e_reduce_rate:
            flag = 'Finish!\n'
            print flag
            break
        else:
            cost_0 = cost_1
            theta_0 = theta_1
            e_0 = e_1
            step += 1

    return theta_1


def line_search(x, y, alpha, theta_0, grad):
    # 不更新梯度，在当前梯度方向上，迭代100次寻找最佳的下一个theta组合点，
    e_0 = get_e(x, theta_0, y)
    cost_0 = get_cost(e_0)
    max_iter, count, a, b = 1000, 0, 0.8, 0.5

    while count < max_iter:
        # 随count增大，alpha减小，0.8*0.8*0.8*0.8*...
        alpha_count = pow(a, count) * alpha
        theta_1 = theta_0 - alpha_count*grad
        e_1 = get_e(x, theta_1, y)
        cost_1 = get_cost(e_1)
        # 当前theta组合下的梯度为grad, grad == tan(w) == 切线y变化量／theta变化量， 推出 切线y变化量 = grad * theta变化量
        grad_dy_chage = abs(np.dot(grad.T, theta_1-theta_0))
        # 实际y变化量为cost0-cost1, 不能加绝对值，如果加了就不收敛了。只有cost_y_chage>0且大于b*grad_dy_chage时才符合条件
        cost_y_change = cost_0-cost_1
        # 当前梯度方向上，实际y变化量 占 切线y变化量的比率为b，b越大越好，至少大于0.5才行，实际损失减小量要占 dy的一半以上。
        if cost_y_change > b*grad_dy_chage:
            break
        else:
            count += 1
    return theta_1, cost_1, e_1, count


def iter_bfgs(x, theta_0, y, out_n, out_e_reduce_rate, dfp):
    e_0 = get_e(x, theta_0, y) # e0 是ypred 和 y之间的距离
    cost_0 = get_cost(e_0)
    grad_0 = np.reshape(compute_grad(x, e_0), (x.shape[1], 1))
    hk = np.eye(x.shape[1])
    print 'hk', hk.shape
    print 'grad', grad_0.shape
    step = 1
    while step < out_n:
        dk = -1*np.dot(hk, grad_0)
        print 'dk', dk.shape
        theta_1, cost_1, e_1, l_count = armijo_line_search(x, y, theta_0, grad_0, dk, cost_0)
        grad_1 = compute_grad(x, e_1)
        # print theta_1
        # print grad_1
        yk = grad_1 - grad_0
        sk = theta_1 - theta_0
        print('sk', sk.shape)

        condition = np.dot(sk.T, yk)
        # 更新以此theta就更新一次hk
        if dfp == 1:
            if condition > 0:
                hk = hk - X2(X3(hk, yk, yk.T), hk)/X3(yk.T, hk, yk)+(X2(sk, sk.T))/condition
        else:
            if condition > 0:
                hk = hk + (1+X3(yk.T, hk, yk)/condition)*(X2(sk, sk.T)/condition)-(X3(sk, yk.T, hk)+X3(hk, yk, sk.T))/condition
        e_reduce_rate = abs(cost_1 - cost_0) / cost_0
        print 'Step: %-6s, l_count: %-6s, cost: %s' % (step, l_count, cost_1[0,0])
        if e_reduce_rate < out_e_reduce_rate:
            flag = 'Finish!\n'
            print flag
            break
        cost_0, grad_0, theta_0 = cost_1, grad_1, theta_1
        step += 1

    return theta_1


def iter_lbfgs(x, theta_0, y, out_n, out_e_reduce_rate):
    limit_n = 5
    ss, yy = [], []

    step = 1
    while step < out_n:

        if step == 1:
            e_0 = get_e(x, theta_0, y)
            cost_0 = get_cost(e_0)
            grad_0 = compute_grad(x, e_0)
            dk = -1*grad_0

        theta_1, cost_1, e_1, l_count = armijo_line_search(x, y, theta_0, grad_0, dk, cost_0)
        grad_1 = compute_grad(x, e_1)

        if len(ss) > limit_n and len(yy) > limit_n:
            del ss[0]
            del yy[0]

        yk = grad_1 - grad_0
        sk = theta_1 - theta_0
        ss.append(sk)
        yy.append(yk)

        qk = grad_1
        k = len(ss)
        condition = X2(yk.T, sk)
        alpha = []

        for i in range(k):
            # t 4->0 倒着计算，倒着存alpha
            t = k-i-1
            pt = 1 / X2(yy[t].T, ss[t])
            alpha.append(pt*X2(ss[t].T, qk))

            qk = qk - alpha[i] * yy[t]

        z = qk

        for i in range(k):
            t = k - i - 1
            # i 0->4 正着计算，正着存beta
            pi = 1 / (X2(yy[i].T, ss[i])[0, 0])
            # pi数
            beta = pi*X2(yy[i].T, z)
            # beta[i]数
            z = z+ss[i]*(alpha[t] - beta)

        if condition > 0:
            dk = -z

        e_reduce_rate = abs(cost_1 - cost_0) / cost_0
        print 'Step: %-6s, l_count: %-6s, cost: %s' % (step, l_count, cost_1[0, 0])
        if e_reduce_rate < out_e_reduce_rate:
            flag = 'Finish!\n'
            print flag
            break
        cost_0, grad_0, theta_0 = cost_1, grad_1, theta_1
        step += 1

    return theta_1


def armijo_line_search(x, y, theta_0, grad_0, dk_0, cost_0):
    # print theta_0.shape, grad_0.shape, dk_0.shape, cost_0.shape
    # 不更新梯度，在当前梯度方向上，迭代100次寻找最佳的下一个theta组合点，
    max_iter, count, countk, a, b = 100, 0, 0, 0.55, 0.4

    while count < max_iter:
        """
        batch方法使用梯度方向grad更新theta，newton等使用牛顿方向dk来更新theta
        newton:hessian逆*grad; dfp:当前步dfpHk; bfgs:当前步bfgsHk
        当前梯度方向上，实际y变化量 占 切线y变化量的比率为b，b越大越好，至少大于0.5才行，实际损失减小量要占 dy的一半以上。
        """

        """更新theta"""
        theta_1 = theta_0 + pow(a, count)*dk_0
        """计算损失"""

        e_1 = get_e(x, theta_1, y)
        cost_1 = get_cost(e_1)
        cost_y_change = cost_1 - cost_0
        dy_change = b * pow(a, count) * np.dot(grad_0.T, dk_0)

        # if cost_1 < cost_0 + b * pow(a, count) * np.dot(grad_0.T, dk_0):
        if cost_y_change < dy_change:
            """如果一直不满足条件，那么一直没有将count赋值给countk，countk仍为0"""
            countk = count
            break
        count += 1

    theta_1 = theta_0 + pow(a, countk) * dk_0
    e_1 = get_e(x, theta_1, y)
    cost_1 = get_cost(e_1)
    return theta_1, cost_1, e_1, count


if __name__ == '__main__':

    # x1, y1 = get_data('/optdata/line_sample_data.txt')
    N = 500
    D = 300
    x1 = np.random.randn(N, D)
    y1 = np.random.randn(N, 1)
    # x1, y1 = get_data('/optdata/airfoil_self_noise.txt')
    theta1 = np.zeros(x1.shape[1]).reshape(x1.shape[1], 1)

    # res_theta = iter_batch(x1, theta1, y1, out_n=1e5, out_e_reduce_rate=1e-6, alpha=0.02)
    # res_theta = iter_random_batch(x1, theta1, y1, out_n=1e5, out_e_reduce_rate=1e-6, alpha=0.02)
    # res_theta = iter_mini_batch(x1, theta1, y1, out_n=1e5, out_e_reduce_rate=1e-6, alpha=0.01, batch=10)
    # res_theta = iter_batch_linesearch(x1, theta1, y1, out_n=1e5, out_e_reduce_rate=1e-6, alpha=1)
    # res_theta = iter_newton(x1, theta1, y1, out_n=1e5, out_e_reduce_rate=1e-6, alpha=1)

    res_theta = iter_bfgs(x1, theta1, y1, out_n=100, out_e_reduce_rate=1e-6, dfp=1)
    # res_theta = iter_bfgs(x1, theta1, y1, out_n=100, out_e_reduce_rate=1e-6, dfp=0)
    # res_theta = iter_lbfgs(x1, theta1, y1, out_n=100, out_e_reduce_rate=1e-6)
    # print 'Res_theta:', np.reshape(res_theta, (1, res_theta.shape[0]))[0]

    # def iter_bfgs(x, theta_0, y, out_n, out_e_reduce_rate):

  grad_diff = wg_k_1 - wg_k
  sk = wk_1 - wk
  ss.append(sk)
  yy.append(grad_diff)

  qk = wg_k_1
  k = len(ss)
  condition = grad_diff.T@sk
  alpha = []

  for j in range(k):
      t = k-j-1
      pt = 1 / (yy[t].T@ss[t])
      alpha.append(pt*(ss[t].T@qk))

      qk = qk - alpha[j] * yy[t]

  z = qk

  for j in range(k):
      t = k - j - 1
      pj = 1 / ((yy[j].T@ss[j])[0, 0])
 
      beta = pj*(yy[j].T@z)
     
      z = z+ss[j]*(alpha[t] - beta)

  if condition > 0:
      dk = z
  return z