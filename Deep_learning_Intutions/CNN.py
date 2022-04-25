import numpy as np 
import matplotlib.pyplot as plt
import pandas 

np.random.seed(4)


def zero_pad(X,pad):
    """
        args:
            X -> batch of m images in numpy array of shape (m,n_H,n_W,n_C)
            pad -> amount of padding around each image on vertical and horizontal directions


        Returns:
            X_pad -> padded image of shape (m,nH + 2 * pad, n_W + 2 * pad, n_C)
    """

    X_pad = np.pad(X,((0,0),(pad,pad), (pad,pad),(0,0)))

    return X_pad

def cov_single_step(a_slice_prev,W,b):
    s = np.multiply(a_slice_prev,W) #element wise product
    z = np.sum(s)
    b = np.squeeze(b)
    z = z + b

    return zero_pad

