from numpy import *

def find_knn(x, data, k, norm):
    "Return indices in DATA of the k nearest neighbors to X, "
    " as measured by NORM function."
    x = x.reshape((size(a), 1))
    distances = array(map(lambda y: norm(x-y), data))
    return argsort(distances)[:k]

def knn_classify(x, data, y, k=None, norm=linalg.norm):
    "Return KNN classification of X. Default to k = sqrt(n)."
    if k == None: k = ceil(sqrt(data.shape[1]))
    knn = find_knn(x, data, k, norm)
    classes = bincount(array(map(lambda i: y[i], knn)))
    return classes.argmax()
