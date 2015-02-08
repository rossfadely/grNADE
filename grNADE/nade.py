import numpy as np
from utils import softmax
from optimization import optimize

class OrderedNADE(object):
    """
    An implementation of http://arxiv.org/abs/1306.0186, for now restricted to
    the case of continuous valued density estimation with MOGs.
    """
    def __init__(self, data, learning_rate=0.1, momentum=0.9,
                 Nhidden=128, Nlayer=1, Ncomponents=8, Nepochs=100,
                 slow_factor = 0.1, Verbose=True):
        self.Ndim = dim.shape[1]
        self.Ndata = data.shape[0]
        self.Nlayer = Nlayer
        self.Nhidden = Nhidden
        self.Ncomponents = Ncomponents
        self.slow_factor = slow_factor

        self.check_init()
        self.param_init()
        self.run(data, Nepochs, learning_rate, momentum, verbose)

    def check_init(self):
        """
        For current implementation, check inputs are sensible.
        """
        if self.Nlayer > 1:
            raise Exception("Nlayer == 1 currently")

    def param_init(self, sig=0.01):
        """
        Initialize the parameters of the model.
        """
        self.rhos = np.ones(self.Ndim)
        self.a = np.random.rand(self.Ndim, self.Ndim)
        self.c = np.random.rand(self.Ndim)
        self.W = np.random.randn(self.Nhidden, (self.Ndim - 1)) * sig
        self.alphas = np.zeros((self.Ndim, self.Ncomponents))
        self.mus = np.zeros((self.Ndim, self.Ncomponents))
        self.vars = np.zeros((self.Ndim, self.hidden, self.Ncomponents))

        types = ['alpha', 'mu', 'sigma']
        self.bs = {}
        self.Vs = {}
        for t in types:
            self.bs[t] = np.random.randn(self.Ndim, self.Ncomponents) * sig
            self.Vs[t] = np.random.randn(self.Ndim, self.Nhidden,
                                         self.Ncomponents) * sig

    def eval_nll(self, data):
        """
        Evaluate the negative log likelihood for a single datum.
        """
        nll = 0.0
        self.a[0] = self.c
        for i in range(self.Ndim):
            a = self.rhos[i] * self.a[i]
            h = 0.5 * (a + np.abs(a)) # ReLU
            za = np.dot(self.Vs['alpha'][i].T, h) + self.bs['alpha']
            zm = np.dot(self.Vs['mu'][i].T, h) + self.bs['mu']
            zs = np.dot(self.Vs['sigma'][i].T, h) + self.bs['sigma']
            self.alphas[i] = softmax(za)
            self.mus[i] = zm
            self.vars[i] = np.exp(zs)
            nll -= nll_mog(data[i], self.alphas[i], self.mus[i],
                           self.sigmas[i])
            if i > 0:
                self.a[i - 1] += np.dot(data[i], self.W[:, i])
        return nll

    def eval_grads(self, data):
        """
        Calculate gradients for the model given a datum.
        """
        drho = np.zero_like(self.rhos)
        dW = np.zeros_like(self.W)
        da = np.zeros_like(self.a)
        dc = np.zeros_like(self.c)
        dba = np.zeros_like(self.bs['alpha'])
        dVa = np.zeros_like(self.Vs['alpha'])
        dbm = np.zeros_like(self.bs['mu'])
        dVm = np.zeros_like(self.Vs['mu'])
        dbs = np.zeros_like(self.bs['sigma'])
        dVs = np.zeros_like(self.Vs['sigma'])

        const = 0.5 * np.log(2. * np.pi)
        for i in range(self.Ndim, 0, -1):
            a = self.rhos[i] * self.a[i]
            h = 0.5 * (a + np.abs(a)) # ReLU
            dlt = (self.data - self.mus[i]) / self.vars[i]
            phi = 0.5 * dlt ** 2. - np.log(self.vars[i]) - const
            pi = self.alphas[i] * phi
            pi /= np.sum(pi)
            dza = pi - self.alphas[i]
            dVa[i] = dza * h
            dzm = pi * dlt
            dzm *= self.slow_factor # apparently this is a `tight' component
            dVm[i] = dzm * h
            dbm[i] = dzm
            dzs = pi * (dlt ** 2. - 1)
            dVs[i] = dzs * h
            dbs = dzs
            dh = dza * dVa[i] + dzm * dVm[i] + dzs * dVs[i]
            dpsi = 1. * (dh > 0)
            drho[i] = np.sum(dpsi)
            da[i] = da[i] + dpsi * self.rho[i]
            dW[:, i] = da[i] * data[i]
            if i == 0:
                dc = da[i]
        return drho, dW, dc, dba, dVa, dbm, dVm, dbs, dVs

    def run(self, Nepochs, learning_rate, momentum, verbose):
        """
        Optimize the model.
        """
        optimize(self, data, Nepochs, learning_rate, momentum, verbose)
