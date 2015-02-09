import numpy as np

def softmax(x):
    """
    1D softmax function
    """
    v = np.exp(x - x.max())
    return v / np.sum(v)

def log_normal_densities_1D(x, alphas, mus, sigmas):
    """
    Return log likelihood for 1D gaussians.
    """
    return (x[None, :] - mus)

def loglike_MOG_1D(x, alphas, mus, sigmas, responsibilities=False):
    """
    Return the log likelihoods of the data given the MOG params.
    """
    log_probs = -0.5 * ((x[:, None] - mus[None, :]) / sigmas[None, :]) ** 2.
    log_probs += -0.5 * (np.log(2. * np.pi) + np.log(sigmas[None, :] ** 2.))
    log_probs += np.log(alphas[None, :])
    tot_log_probs = logsumexp(log_probs, axis=1)
    if responsibilities:
        rs = np.exp(log_probs - tot_log_probs[:, None])
    else:
        rs = None
    return tot_log_probs, rs

def nll_MOG_1D(x, alphas, mus, sigmas, responsibilities=False):
    """
    Return the negative log likelihood of a mixtures of gaussians.
    """
    ll, rs = loglike_MOG_1D(x, alphas, mus, sigmas, responsibilities)
    return -1. * ll, rs

def logsumexp(arr, axis=None):
    """
    Swiped from astroML:
    https://github.com/astroML/astroML/blob/master/astroML/utils.py
    
    Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
      """
    # if axis is specified, roll axis to 0 so that broadcasting works below
    if axis is not None:
        arr = np.rollaxis(arr, axis)
        axis = 0

    # Use the max to normalize, as with the log this is what accumulates
    # the fewest errors
    vmax = arr.max(axis=axis)
    out = np.log(np.sum(np.exp(arr - vmax), axis=axis))
    out += vmax

    return out

def sample_MOG_1D(alpha, mu, V, size=1):
    alpha_cs = np.cumsum(alpha)
    r = np.atleast_1d(np.random.random(size))
    r.sort()

    ind = r.searchsorted(alpha_cs)
    ind = np.concatenate(([0], ind))
    if ind[-1] != size:
        ind[-1] = size

    draws = np.array([])
    for i in range(alpha.size):
        draws = np.append(draws, np.random.normal(mu[i], V[i],
                                                  (ind[i + 1] - ind[i])))
    draws = draws[np.random.permutation(size)]
    return draws
