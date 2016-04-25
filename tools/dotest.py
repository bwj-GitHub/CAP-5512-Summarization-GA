"""
Created on Apr 22, 2016

@author: brandon
"""


import numpy as np
from scipy.spatial.distance import cosine
import random


def xover(p1, p2):
    w_shape = p1.shape
    prop = .5  # random.random()
    mask1 = np.random.choice(a=[0,1],
                             size=w_shape,
                             p=[prop, 1-prop])
    #mask2 = np.ones(*w_shape) - mask1
    mask2 = np.ones(w_shape) - mask1

    c1 = mask1 * p1 + mask2 * p2
    c2 = mask1 * p2 + mask2 * p1
    return c1, c2


def dropoff(W, p=.5):
    """Randomly set values in W to 0 with p probability each."""

    DO = np.random.choice(a=[0,1], size=W.shape, p=[p, 1-p])
    return W*DO


def test_dot1(a=10, b=10, c=10, d=1, runs=1000, sparsify=True, sp=.8):
    # (a,b) [dot] (c,d) >> (a,d)
    # Generate parents with dims (a,b):
    p1 = np.random.randn(a, b)
    p2 = np.random.randn(a, b)
    o = np.random.randn(a, b)
    
    if sparsify:
        p1 = dropoff(p1, sp)
        p2 = dropoff(p2, sp)
        o = dropoff(o, sp)
    
    # Create children with 'Xover':
    c1, c2 = xover(p1, p2)

    a = [0, 0, 0, 0, 0, 0]  # average sim: p1c1, p1c2, p2c1, p2c2, oc1, oc2
    # Compare outputs for randomly generated inputs:
    for i in range(runs):
        x = np.random.randn(c, d)
        y0 = np.dot(p1, x)
        y1 = np.dot(p2, x)
        yo = np.dot(o, x)
        y2 = np.dot(c1, x)
        y3 = np.dot(c2, x)
        a[0] += 1 - cosine(y0, y2)
        a[1] += 1 - cosine(y0, y3)
        a[2] += 1 - cosine(y1, y2)
        a[3] += 1 - cosine(y1, y3)
        a[4] += 1 - cosine(yo, y2)
        a[5] += 1 - cosine(yo, y3)
    for i in range(len(a)):
        a[i] /= runs
    print(str(a[0:4]) + "\n" + str(a[4:]))


if __name__ == "__main__":
    test_dot1()
    print("finished")
