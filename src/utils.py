import numpy as np


def sinc(x):
    if x == 0.0: return 1;
    else:
        return np.sin(np.pi*x)/(np.pi*x)

def f1(X):
    return sinc(X[0]) * sinc(X[1])

def f2(X):
    out = X[0]**2 + X[1]**2
    out += 2*X[0]*X[1]*np.cos(np.pi*X[0]*X[1])
    out += X[0] + X[1] - 1
    return out

def generate_data(n, low, high, func):
    """
        Generate data from a uniform distribution between a real numbers
        interval as a network features, and generate labels using a function
        passed as a parameter.
    """
    samples = []
    labels = []
    for i in range(n):
        # dealing with division by zero
        if i < 100:
            samples.append(np.array([0.0, 0.0]))
            labels.append(func(samples[-1]))
        else:
            samples.append(np.random.uniform(low, high, (2,)))
            labels.append(func(samples[-1]))

    X = np.vstack(samples)
    y = np.vstack(labels)
    return X, y
