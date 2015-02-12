import numpy as np
import matplotlib.pyplot as pl

from nade import OrderedNADE
from utils import nll_MOG_1D, sample_MOG_1D
from optimization import Optimize
from sklearn.mixture import GMM

def simple_ordered_NADE():
    """
    Run nade on 2D fake data.
    
    The data structure is pulled from http://bit.ly/1FDQdDd
    """
    N = 2000
    np.random.seed(0)

    # generate the true data
    x_true = (1.4 + 2 * np.random.random(N)) ** 2
    y_true = 0.1 * x_true ** 2

    # add scatter to "true" distribution
    dx = 0.1 + 4. / x_true ** 2
    dy = 0.1 + 10. / x_true ** 2
    
    x_true += np.random.normal(0, dx, N)
    y_true += np.random.normal(0, dy, N)

    # add noise to get the "observed" distribution
    dx = 0.2 + 0.5 * np.random.random(N)
    dy = 0.2 + 0.5 * np.random.random(N)

    x = x_true + np.random.normal(0, dx)
    y = y_true + np.random.normal(0, dy)

    # stack the results for computation
    X = np.vstack([x, y]).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([dx ** 2, dy ** 2]).T

    # run NADE
    model = OrderedNADE(X)
    ll_nade = -1 * model.nll(X)

    # run GMM
    gmm = GMM(n_components=20)
    gmm.fit(X)
    ll_gmm = gmm.score(X)
    print 'GMM neg log like:', -1 * np.sum(ll_gmm)

    assert 0, 'Need to sort this out when fresh.'
    # this is stupid but I am tired
    N = 200
    x = np.linspace(X[:, 0].min(), X[:, 0].max(), N)
    y = np.linspace(X[:, 1].min(), X[:, 1].max(), N)
    grid = np.zeros((N ** 2, 2))
    for i in range(N):
        for j in range(N):
            grid[i * 200 + j] = np.array([x[i], y[i]])

    ll_nade_grid = -1 * model.nll(grid).reshape(N, N)
    ll_gmm_grid = gmm.score(grid).reshape(N, N)

    f = pl.figure(figsize=(10, 5))
    pl.subplot(121)
    i = pl.imshow(ll_gmm_grid, origin='lower', interpolation='nearest')
    i.set_cmap('Greys')
    pl.colorbar()
    pl.subplot(122)
    i = pl.imshow(ll_nade_grid, origin='lower', interpolation='nearest')
    i.set_cmap('Greys')
    pl.colorbar()
    f.savefig('../plots/simple_nade_test.png')

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
            grads = np.mean(grads, axis=0)
            return [grads]
            
        def save(self, *args):
            pass

        def optimize_mus(self, data):
            Optimize(data, self, momentum=None, learning_rate=0.03,
                     check_epoch=5, Neval=10, tol=1.e-2)

    mog1d()

if __name__ == '__main__':
    if False:
        SGD_MOG_1D()
    if True:
        simple_ordered_NADE()
