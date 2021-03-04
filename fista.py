import tensorflow as tf


def fista(loss_fn, prox_fn, x0, max_iter=2000, max_linesearch=100, eta=0.5, tol=1e-6,
      verbose=1):
    y = x0
    x = y
    tau = 1.0
    t = 1.0
    for it in range(max_iter):
        f_old, grad = loss_fn(y, True)
        for ls in range(max_linesearch):
            y_proj = prox_fn(y - grad *tau, tau)
            diff = y_proj - y
            sqdist = tf.matmul(diff,diff,transpose_a=True)
            dist = tf.sqrt(sqdist)
            F = loss_fn(y_proj)
            Q = f_old + tf.matmul(diff, grad,transpose_a=True) + 0.5 * sqdist/tau
            if F <= Q:
                print("break")
                break
            tau *= eta
        if ls == max_linesearch - 1 and verbose:
            print("Line search did not converge.")
        if verbose:
            print("%d. %f" % (it + 1, dist))
        if dist <= tol:
            if verbose:
                print("Converged.")
            break
        x_next = y_proj
        t_next = (1 + tf.sqrt(1 + 4 * t ** 2)) / 2.
        y = x_next + (t-1) / t_next * (x_next - x)
        t = t_next
        x = x_next
    return y_proj