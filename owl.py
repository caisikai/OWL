import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import shape



def generate_data(N,K):
    
    rho=np.zeros([K,K])
    block=np.ones([K//3,K//3])
    rho[:K//3,:K//3]+=block*0.9
    rho[K//3:2*K//3,K//3:2*K//3]+=block*0.9
    rho[2*K//3:,2*K//3:]+=block*0
    i=np.arange(K)
    rho[i,i]=1
    rho=tf.linalg.cholesky(tf.constant(rho,dtype=tf.float32))
    simX=tf.matmul(tf.random.uniform(shape=[N,K]),rho)
    
    b=np.random.uniform(size=[K,1])
    b[K//3:2*K//3,:]=0
    

    e=tf.random.uniform(shape=[N,1])*0.1

    y=tf.matmul(simX,tf.constant(b,dtype=tf.float32))+e

    return y,simX,b

def generate_data1(n_samples = 10,n_features = 100):
    
    
    coef = np.zeros(n_features)
    coef[int(0.2*n_features):int(0.3*n_features)] = -1
    coef[int(0.6*n_features):int(0.7*n_features)] = 1
    coef /= np.linalg.norm(coef)
    
    rng = np.random.RandomState(1)
    X = rng.randn(n_samples, n_features)
    X[:, int(0.2*n_features):int(0.3*n_features)] = X[:, int(0.2*n_features)]
    X[:, int(0.6*n_features):int(0.7*n_features)] = X[:, int(0.2*n_features)]
    X += 0.001 * rng.randn(n_samples, n_features)
    X /= np.linalg.norm(X, axis=0)
    y = np.dot(X, coef)

    return tf.expand_dims(tf.constant(y,dtype=tf.float32),axis=1),\
        tf.constant(X,dtype=tf.float32),\
        tf.expand_dims(tf.constant(coef,dtype=tf.float32),axis=1)





class OwlRegressor():

    def __init__(self, weights,max_iter=500,
                 max_linesearch=20, eta=2.0, tol=1e-3,verbose=1):
        self.weights = weights
        self.max_iter = max_iter
        self.max_linesearch = max_linesearch
        self.eta = eta
        self.tol = tol
        self.verbose=verbose

    def loss(self,Y,X,coef,grad=False):
        Y_hat=tf.matmul(X,coef)
        diff=Y_hat-Y
        loss=0.5*tf.reduce_sum(tf.math.square(diff))
        if grad:
            grads=tf.matmul(X,diff,transpose_a=True)           
            return loss,grads
        else:
            return loss

    def prox_owl(self,b,w):
        """Proximal operator of the OWL norm dot(w, reversed(sort(v)))

        Follows description and notation from:
        X. Zeng, M. Figueiredo,
        The ordered weighted L1 norm: Atomic formulation, dual norm,
        and projections.
        eprint http://arxiv.org/abs/1409.4271
        """
        # wlog operate on absolute values
        b=tf.squeeze(b,axis=1)
        b_abs=tf.math.abs(b)
        ix=tf.argsort(b_abs,direction="DESCENDING")
        b_abs=tf.sort(b_abs,direction="DESCENDING")

        # project to K+ (monotone non-negative decreasing cone)
        b_abs,_=tf.nn.isotonic_regression(b_abs-w)

        #undo sort
        inv_ix = np.zeros_like(ix)
        inv_ix[ix.numpy()] = np.arange(len(b))
        v_abs=tf.gather(b_abs,inv_ix)

        return tf.expand_dims(tf.sign(b)*v_abs,axis=1)

    

    
    def fit(self,Y, X,  max_iter=2000, max_linesearch=100, eta=0.5, tol=1e-6,
          verbose=1):

        x0=tf.zeros(shape=[X.shape[1],1])
        y = x0
        x = y
        tau = 1.0
        t = 1.0

        for it in range(max_iter):
            f_old, grad = self.loss(Y,X,y, True)

            for ls in range(max_linesearch):
                y_proj = self.prox_owl(y - grad *tau, self.weights*tau)
                diff = y_proj - y
                sqdist = tf.matmul(diff,diff,transpose_a=True)
                dist = tf.sqrt(sqdist)

                F = self.loss(Y,X,y_proj)
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
            t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
            y = x_next + (t-1) / t_next * (x_next - x)
            t = t_next
            x = x_next

        self.coef=y_proj


def fista_owl(N=100,K=90):
    y,simX,b=generate_data(N,K)

    alpha = 0.001
    

    import matplotlib.pyplot as plt
    plt.figure()

    # ground truth:
    plt.subplot(221)
    plt.stem(np.arange(K), b)
    plt.title("True coefficients")

    from sklearn.linear_model import Lasso
    plt.subplot(222)
    lasso_skl = Lasso(alpha=alpha / (2 * N), fit_intercept=False)
    lasso_skl.fit(y, simX)
    plt.stem(np.arange(K), lasso_skl.coef_)
    plt.title("LASSO coefficients (scikit-learn)")

    plt.subplot(223)
    owl=OwlRegressor(weights=tf.ones(K)*alpha)
    owl.fit(y,simX)
    plt.stem(np.arange(K), owl.coef)
    plt.title("LASSO coefficients (pyowl)")

    plt.subplot(224)
    plt.stem(np.arange(K), owl.coef-b)
    plt.title("LASSO errors (pyowl)")

    plt.tight_layout()
    plt.savefig(f"my_toy_example{N}.png")

fista_owl(N=100,K=90)
fista_owl(N=1000,K=90)
fista_owl(N=70,K=90)
