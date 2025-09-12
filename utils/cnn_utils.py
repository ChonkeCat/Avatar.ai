#helper functions for the CNN, e.g. loading dataset, ReLU, padding, etc.
import numpy as cp

## Computes the softmax for a 1-D vector
## inputs: x -> 1-D vector, grad -> true if you want the gradiant, false otherwise
## outputs: 1-D vector corresponding to the normalized verison on the input
##          or nxn jacobian of the softmax

def softmax(x, grad = False):
    shiftx = x - cp.max(x) #shifts the values in x, for computational stabillity
    expx = cp.exp(shiftx)
    if not grad:
        return expx / cp.sum(expx)
    x = expx / cp.sum(expx)
    x = x.reshape(-1, 1)
    return cp.diagflat(x) - cp.dot(x, x.T)


## LeakyRelU activation function, returns x if it is greater than or equal to 0, returns 0 otherwise
## inputs: x -> float, grad => true if you want the gradiant, false otherwise

def LeakyRelU(x, grad = False):
    alpha = 0.05
    if not grad:
        return cp.maximum(alpha*x, x)
    if x <= 0:
        return alpha
    return 1

test = cp.array ([-10, -5, 0, 5, 10])

for x in test:
    print(LeakyRelU(x))
    print(LeakyRelU(x, True))