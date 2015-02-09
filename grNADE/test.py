import numpy as np

from utils import nll_MOG_1D, sample_MOG_1D
from optimization import Optimize

def SGD_MOG_1D():
    """
    Test for SGD in 1D MOG case.
    """
    class mog1d(object):
        def __init__(self, alphas=np.array([0.2, 0.8]),
                     mus=np.array([-2.5, 4.5]), sigmas=np.array([0.5, 0.5]),
                     N=5000):

            self.model_mus = np.zeros(2)
            self.optimize_params = [self.model_mus]
            self.alphas = alphas
            self.mus = mus
            self.sigmas = sigmas

            data = self.gen_data(alphas, mus, sigmas, N)
            self.optimize_mus(data)

        def gen_data(self, alphas, mus, sigmas, N):
            return sample_MOG_1D(alphas, mus, sigmas ** 2., N)

        def eval_nll(self, data):
            nlls, self.rs = nll_MOG_1D(data, self.alphas, self.model_mus,
                                       self.sigmas, responsibilities=True)
            return nlls.sum()

        def eval_grads(self, data):
            grads = (data[:, None] - self.model_mus) / self.sigmas ** 2.
            grads *= -self.rs
            return grads
            
        def save(self, *args):
            pass

        def optimize_mus(self, data):
            Optimize(data, self, momentum=None, learning_rate=0.03,
                     check_epoch=5, Neval=10, tol=1.e-2)

    mog1d()

if __name__ == '__main__':
    SGD_MOG_1D()
