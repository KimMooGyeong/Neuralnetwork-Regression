import numpy as np

class DataModule:
    """Class for synthetic data with no error"""
    # initialization
    def __init__(self, x_min=-1.0, x_max=1.0, N=50):
        self._x_min = x_min
        self._x_max = x_max
        self._N = N
        self._step = (x_max - x_min)/(N - 1)

    # extract data set
    def gen_data(self, f):
        X = [self._x_min + self._step * i for i in range(self._N)]
        x_data = np.array(X).reshape((self._N,1))
        y_data = np.array([f(x) for x in X]).reshape(self._N, 1)
        return x_data, y_data

    def get_sample_number(self):
        return self._N

    def set_sample_number(self, n):
        self._N = n
        return

    # f(x) = x**2
    def quadratic(self, x):
        return 2*x**2 - 1

    # f(x) = sin(x)
    def sinusodial(self, x):
        return np.sin(np.pi * x)

    # f(x) = |x|
    def absolute(self, x):
        return 2*abs(x)-1

    # f(x) = step(x)
    def heviside(self, x):
        if x >= 0:
            return 1
        else:
            return -1
