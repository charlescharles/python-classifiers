from numpy import *
from scipy import optimize

def sigmoid(x):
    return 1/(1+ exp(-x))


def cost(theta, X, y, lam):
    m = X.shape[0]
    pred = sigmoid(dot(X, theta))
    c_term = sum(-y*log(pred) - (1-y)*log(1-pred))/m
    reg_term = lam*sum(theta[1:]**2) / (2.0*m)
    return c_term + reg_term


def grad(theta, X, y, lam):
    m = X.shape[0]
    pred = sigmoid(dot(X, theta))
    grad = zeros(theta.shape)
    grad[0] = sum(1.0*pred- y)/m
    grad[1:] = dot(X[:,1:].T, (pred-y)/m) + 1.0*lam*theta[1:]/m
    return grad


def train(X, y, theta0=None, N=1000, lam=1):
    if not theta0:
        theta0 = array([0.0001]*(X.shape[1]+1))
    X = add_bias(X)
    args = (X, y, lam)
    params = (cost, theta_init, args)
    #theta = optimize.fmin(*params, jac=grad, options={'maxiter':N})
    theta = optimize.fmin_bfgs(cost,theta0,fprime=grad, args=(X,y,1)
    return theta


def add_bias(X):
    b = ones((X.shape[0], X.shape[1]+1))
    b[:,1:] = X
    return b

if __name__ == '__main__':
    data =  load('./digit-classification/data0.npy')
    theta0 = train(data, ones((data.shape[0])))
    print(theta0)