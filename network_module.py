import numpy as np
import numpy.linalg as lin
from numpy.random import randn
from scipy.special import expit

class SimpleNetwork:
    """ Class of neural network such that has M hidden units in
        one hidden layer, D input units and one output unit."""
    # initialzation
    def __init__(self, M=3, D=2, etha=0.01):
        self._M = M
        self._D = D
        self._etha = etha
        self._W1 = None
        self._W2 = None
        self._set_weight = False
        self._a1 = None
        self._z1 = None
        self._a2 = None
        self._y = None
        self._gradeE1 = None
        self._gradeE2 = None

    # initialize weight method
    def _init_weight(self):
        self._W1 = (10.0*randn(self._M, self._D)).astype('float64')
        self._W2 = (10.0*randn(self._M)).reshape(1, self._M).astype('float64')
        return self._W1, self._W2

    # initialize gradient E
    def _init_grad(self):
        self._gradE1 = np.zeros((self._M, self._D), dtype=np.float64).reshape((self._M, self._D)).astype('float64')
        self._gradE2 = np.zeros(self._M, dtype=np.float64).reshape((1, self._M)).astype('float64')

    # function definitions
    def _linear(self, x):
        return x

    def _linearp(self, x):
        return 1.0

    def _sig(self, x):
        return expit(x)

    def _tanh(self, x):
        return 2.0 * self._sig(2.0*x) - 1.0

    def _sigp(self, x):
        return self._sig(x) * (1.0 - self._sig(x))

    def _tanhp(self, x):
        tanh = self._tanh(x)
        return 1.0 - (tanh * tanh)

    # forwrad propagation methods
    def _eval_a1(self, x):
        return np.matmul(self._W1, x.T)

    def _eval_z1(self, a1):
        return self._tanh(a1)

    def _eval_a2(self, z1):
        return np.matmul(self._W2, z1)

    def _eval_output(self, a2):
        return self._linear(a2)

    def _forward(self, x):
        self._a1 = self._eval_a1(x)
        self._z1 = self._eval_z1(self._a1)
        self._a2 = self._eval_a2(self._z1)
        self._y = self._eval_output(self._a2)
        return

    # back propagation methods
    def _error1(self, x, t):
        delta = self._y - t
        vec1 = x    # 1 by D vector
        vec2 = self._tanhp(self._a1) * self._W2.T   # M by 1 vector
        cons = delta * self._linearp(self._a2)
        return cons * np.matmul(vec2, vec1) # return M by D matrix

    def _error2(self, x, t):
        delta = self._y - t
        return delta * (self._z1.T)

    # batch gradient descent
    def _batchE(self, x, t):
        # forward propagate
        self._forward(x)
        # back propagate
        error1 = self._error1(x, t)
        error2 = self._error2(x,t)
        return error1, error2

    def _backprop(self, data_set, target):
        i = 0
        one = np.array([1.0])
        self._init_grad()
        # evaluate gradE from gradEn
        n = data_set.shape[0]
        while i < n:
            x = np.concatenate((one, data_set[i]), axis=0).reshape((1, self._D)).astype('float64')
            t = target[i][0]
            error1, error2 = self._batchE(x, t)
            self._gradE1 += error1
            self._gradE2 += error2
            i += 1
        self._W1 -= (self._etha * self._gradE1)
        self._W2 -= (self._etha * self._gradE2)
        return

    def _error_function(self, x, t):
        self._forward(x)
        return (self._y - t)**2 / 2

    def _numeric_diff1(self, x, t, epsilon):
        grE = np.zeros(self._M * self._D).reshape(self._M, self._D)
        for i in range(self._M):
            for j in range(self._D):
                self._W1[i][j] += epsilon
                Ep = self._error_function(x, t)
                self._W1[i][j] -= 2.0 * epsilon
                Em = self._error_function(x, t)
                self._W1[i][j] += epsilon
                grE[i][j] = (Ep - Em)/(2.0 * epsilon)
        return grE

    def _numeric_diff2(self, x, t, epsilon):
        grE = np.zeros(self._M).reshape(1, self._M)
        for i in range(self._M):
            self._W2[0][i] += epsilon
            Ep = self._error_function(x, t)
            self._W2[0][i] -= 2.0 * epsilon
            Em = self._error_function(x, t)
            self._W2[0][i] += epsilon
            grE[0][i] = (Ep - Em)/(2.0 * epsilon)
        return grE

    def _check_corr(self, data_set, target, epsilon):
        n_sample = data_set.shape[0]
        self._init_grad()
        check1, check2 = self._gradE1, self._gradE2
        for n in range(n_sample):
            x = np.concatenate((np.ones(1), data_set[n]), axis=0).reshape(1, self._D)
            t = target[n][0]
            error1, error2 = self._batchE(x, t)
            numeric1 = self._numeric_diff1(x, t, epsilon)
            numeric2 = self._numeric_diff2(x ,t, epsilon)
            check1 += (error1 - numeric1)
            check2 += (error2 - numeric2)
        return check1, check2



    """public methods"""
    # modify the number of hidden units
    def set_h_unit(self, n):
        self._M = n
        return

    def set_learning_rate(self, etha=0.01):
        self._etha = etha
        return

    def set_weight(self, W1, W2):
        self._W1 = W1
        self._W2 = W2
        self._set_weight = True
        return

    # method for learning
    def learn(self, data_set, target, tau):
        i = 0
        epsilon = 1e-9
        if self._set_weight == False:
            self._init_weight()
        W10, W20 = self.get_weight()
        for i in range(tau):
            #if (i+1) % 250 == 0:
                #print(self._check_corr(data_set, target, (i%100)*epsilon))
            self._backprop(data_set, target)
        return W10, W20

    # method to obtain weight
    def get_weight(self):
        return self._W1, self._W2

    # return the output for given data point
    def output(self, x):
        self._forward(x)
        return self._y

    # function to plot the result(2-D regression)
    def f_2D_regression(self, data_set):
        Y = []
        i = 0
        n_sample = data_set.shape[0]
        while i < n_sample:
            x = np.concatenate((np.ones(1), data_set[i]), axis=0).reshape((1, self._D))
            y = self.output(x)[0][0]
            Y.append(y)
            i += 1
        return Y

    # funcion to plot the output values of M hidden units
    def z1_2D_regression(self, data_set):
        n_sample = data_set.shape[0]
        dummy = np.ones(self._M).reshape((1, self._M))
        for i in range(n_sample):
            x = np.concatenate((np.ones(1), data_set[i]), axis=0).reshape(1, self._D)
            self._forward(x)
            z = self._z1.T
            dummy = np.r_[dummy, z]
        return dummy[1:, :]

    # method to evaluate error rate
    def error_rate(self, data_set, target):
        err = 0
        i = 0
        n_sample = data_set.shape[0]
        while i < n_sample:
            x = np.concatenate((np.ones(1), data_set[i]), axis=0).reshape((1, self._D))
            y = self.output(x)[0][0]
            t = target[i][0]
            err += (t - y)**2.0
            i += 1
        return err*100/n_sample
