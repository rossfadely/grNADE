import numpy as np
from utils import softmax, nll_MOG_1D
from optimization import Optimize

class OrderedNADE(object):
    """
    An implementation of http://arxiv.org/abs/1306.0186, for now restricted to
    the case of continuous valued density estimation with MOGs.
    """
    def __init__(self, data, learning_rate=1e-6, momentum=0.9,
                 Nhidden=128, Nlayer=1, Ncomponents=8, Nepochs=100,
                 slow_factor = 0.1, verbose=True):
        self.Ndim = data.shape[1]
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
        self.a = np.random.rand(self.Ndim, self.Nhidden)
        self.c = np.random.rand(self.Nhidden)
        self.W = np.random.randn(self.Nhidden, self.Ndim) * sig
        self.alphas = np.zeros((self.Ndim, self.Ncomponents))
        self.mus = np.zeros((self.Ndim, self.Ncomponents))
        self.sigmas = np.zeros((self.Ndim, self.Ncomponents))
        self.optimize_params = [self.rhos, self.c, self.W]

        types = ['alpha', 'mu', 'sigma']
        self.bs = {}
        self.Vs = {}
        for t in types:
            self.bs[t] = np.random.randn(self.Ndim, self.Ncomponents) * sig
            self.Vs[t] = np.random.randn(self.Ndim, self.Nhidden,
                                         self.Ncomponents) * sig
            self.optimize_params.append(self.bs[t])
            self.optimize_params.append(self.Vs[t])

    def nll(self, data):
        """
        Evaluate the negative log likelihood for a batch of data.
        """
        nll = np.zeros(data.shape[0])
        self.a[0] = self.c
        for i in range(self.Ndim):
            a = self.rhos[i] * self.a[i]
            h = 0.5 * (a + np.abs(a)) # ReLU
            za = np.dot(self.Vs['alpha'][i].T, h) + self.bs['alpha'][i]
            zm = np.dot(self.Vs['mu'][i].T, h) + self.bs['mu'][i]
            zs = np.dot(self.Vs['sigma'][i].T, h) + self.bs['sigma'][i]
            self.alphas[i] = softmax(za)
            self.mus[i] = zm
            self.sigmas[i] = np.exp(zs)
            self.vars = self.sigmas ** 2.
            nll += nll_MOG_1D(data[:, i], self.alphas[i], self.mus[i],
                             self.vars[i])[0]
        return nll

    def eval_nll(self, data):
        """
        Return the total negative log likelihood for a batch of data.
        """
        return np.sum(self.nll(data))

    def eval_grads(self, data):
        """
        Calculate gradients for the model given a batch of data..
        """
        drho = np.zeros_like(self.rhos)
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
        for i in range(self.Ndim - 1, -1, -1):
            a = self.rhos[i] * self.a[i]
            h = 0.5 * (a + np.abs(a)) # ReLU
            dlt = (data[:, i][:, None] - self.mus[i][None, :])
            dlt /= self.sigmas[i][None, :]
            dlt = np.mean(dlt, axis=0)
            phi = 0.5 * dlt ** 2. - np.log(self.sigmas[i]) - const
            pi = self.alphas[i] * phi
            pi /= np.mean(pi)
            dza = pi - self.alphas[i]
            dba[i] = dza
            dVa[i] = dza[None, :] * h[:, None]
            dzm = pi * dlt
            dzm *= self.slow_factor # apparently this is a `tight' component
            dbm[i] = dzm
            dVm[i] = dzm[None, :] * h[:, None]
            dzs = pi * (dlt ** 2. - 1)
            dbs[i] = dzs
            dVs[i] = dzs[None, :] * h[:, None]

            # dh has shape Nhidden x Ncomponents (?)
            dh = dza * dVa[i] + dzm * dVm[i] + dzs * dVs[i]
            dpsi = 1. * (dh > 0)

            # collapse to a scalar or vector of Nhidden?
            drho[i] = np.mean(dpsi)

            if i == 0:
                dc = da[i]
            else:
                da[i - 1] = da[i] + np.mean(dpsi * self.rhos[i], axis=1)
                dW[:, i] = np.mean(da[i - 1][:, None] * data[:, i][None, :],
                                   axis=1)
                self.a[i - 1] = self.a[i] - np.mean(data[:, i][None, :] *
                                                    self.W[:, i][:, None],
                                                    axis=1)

        return -drho, -dc, -dW, -dba, -dVa, -dbm, -dVm, -dbs, -dVs

    def run(self, data, Nepochs, learning_rate, momentum, verbose):
        """
        Optimize the model.
        """
        Optimize(data, self, None, Nepochs=Nepochs,
                 learning_rate=learning_rate, momentum=momentum,
                 verbose=verbose)
    
    def save(self, *args):
        """
        Save parameters and optimization proceeds.
        """
        # need to do!!
        pass
