import numpy as np

class Optimize(object):
    """
    Optimization using SGD with momentum.  Model should have an attribute
    called `optimize_params' which is a list of the parameters to be optimized.
    This list should be in the same order as returned by the gradient function.
    In addition, there should be a `eval_cost' and `eval_grads' which return
    the value of the cost and the gradients for a single datum, respectively.
    """
    def __init__(self, data, model, validation_data=None, Nepochs=500,
                 Nbatch=100, learning_rate=0.1,
                 momentum=0.9, check_epoch=10, drop_learning_rate=None,
                 Neval=20, tol=1.e-5,
                 Nthreads=1, verbose=True):

        self.tol = tol
        self.data = data
        self.Ndata = data.shape[0]
        self.model = model
        self.Nparams = len(model.optimize_params)
        self.learning_rate = learning_rate
        self.prev_block_cost = np.inf

        if validation_data is None:
            self.validate = False
            self.valid_costs = np.inf + np.zeros(Nepochs)

        if momentum is None:
            self.update = self.SGD_update
            self.velocities = None
        else:
            self.update = self.SGD_with_momentum_update
            self.momentum = momentum
            self.velocities = {}
            for i, param in enumerate(model.optimize_params):
                self.velocities[i] = np.zeros_like(param)

        self.check_init(Nbatch, Nthreads, drop_learning_rate)
        self.optimize(Nbatch, Nepochs, Neval, check_epoch, verbose)

    def check_init(self, Nbatch, Nthreads, drop_learning_rate):
        if Nbatch > self.Ndata:
            assert False, 'Batch size is bigger than number of samples!'
        if Nthreads > 1:
            raise Exception('Multiprocessing not implemented.')
        if drop_learning_rate is not None:
            raise Exception('Hard (dropping) weight decay not implemented.')

    def SGD_update(self, batch):
        """
        Standard SGD update.
        """
        gradients = self.model.eval_grads(batch)
        for i, param in enumerate(self.model.optimize_params):
            param -= gradients[i] * self.learning_rate

    def SGD_with_momentum_update(self, batch):
        """
        SGD with momentum.
        """
        gradients = self.model.eval_grads(batch)
        for i, param in enumerate(self.model.optimize_params):
            self.velocities[i] *= self.momentum
            self.velocities[i] -= gradients[i] * self.learning_rate
            param += self.velocities[i]

    def optimize(self, Nbatch, Nepochs, Neval, check_epoch, verbose):
        """
        Optimize the model
        """
        done = False
        train_costs = np.zeros(Nepochs) + np.inf
        valid_costs = np.zeros(Nepochs) + np.inf
        for i in range(Nepochs):

            ind = (i * Nbatch) % self.Ndata
            batch = self.data[ind:ind + Nbatch]

            # get costs
            train_costs[i] = self.model.eval_nll(batch)
            block_of_costs = train_costs[i - 2 * Neval:i]
            if self.validate:
                valid_costs = self.model.eval_nll(self.validation_data)
                block_of_costs = valid_costs[i - 2 * Neval:i]

            # move parameters
            self.update(batch)

            # perform monitoring
            if i % check_epoch == 0:
                self.model.save(train_costs, valid_costs, self.velocities)
                if i > 2 * Neval:
                    done = self.check_if_done(block_of_costs, Neval)
                if verbose:
                    print '#' * 80 + '\n'
                    m = 'Epoch:%d, Train cost: %0.3e, Valid cost: %0.3e'
                    m = m % (i, train_costs[i], valid_costs[i])
                    print m + '\n' * 2 + '#' * 80 + '\n' * 2
                if done:
                    break

    def check_if_done(self, block_of_costs, Neval):
        """
        Check if model is done, if so save params
        """
        prev = np.median(block_of_costs[:Neval])
        curr = np.median(block_of_costs[Neval:])
        return (np.abs(prev - curr) / prev) < self.tol
