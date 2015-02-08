import cPickle
import multiprocessing
import numpy as np

class Optimize(object):
    """
    Optimization using SGD with momentum.  Model should have an attribute
    called `optimize_params' which is a list of the parameters to be optimized.
    This list should be in the same order as returned by the gradient function.
    In addition, there should be a `eval_cost' and `eval_grads' which return
    the value of the cost and the gradients for a single datum, respectively.
    """
    def __init__(self, data, model, validation_data=None, Nepoches=100,
                 learning_rate=0.1,
                 momentum=0.9, check_epoch=10, drop_learning_rate=None,
                 Neval=20, tol=1.e-3,
                 Nthreads=1, save_params=(20, 'model'), verbose=True):
        self.tol = tol
        self.data = data
        self.Ndata = data.shape[0]
        self.Neval = Neval
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
            self.velocities= np.zeros(len(model.optimize_params))

        self.check_init(Nthreads, drop_learning_rate)
        self.optimize(Nepochs, save_params, verbose)

    def check_init(self):
        if Nthreads > 1:
            raise Exception('Multiprocessing not implemented.')
        if drop_learning_rate is not None:
            raise Exception('Hard (dropping) weight decay not implemented.')
        if self.Ndata < 2:
            raise Exception('Mini batch needs more than one data point')

    def SGD_update(self):
        """
        Standard SGD update.
        """
        updates = gradients_update()
        for param in self.model.optimize_params:
            param -= updates

    def SGD_with_momentum_update(self):
        """
        SGD with momentum.
        """
        grad_updates = gradients_update()
        for i, param in enumerate(self.model.optimize_params):
            self.velocities[i] *= self.momentum
            self.velocities[i] -= grad_updates
            param += self.velocities[i]

    def gradients_update(self):
        """
        Calculate the gradient part of updates.
        """
        updates = self.learning_rate * self.model.eval_grads(self.data)
        updates = updates.mean(axis=0)
    
    def optimize(self, Nepochs, Neval, verbose):
        """
        Optimize the model
        """
        train_costs = np.zeros(Nepochs) + np.inf
        for i in range(Nepochs):

            # get costs
            train_costs[i] = self.model.eval_nll(self.data)
            block_of_costs = self.train_costs[i - 2 * Neval:i]
            if self.validate:
                valid_costs = self.model.eval_nll(self.validation_data)
                block_of_costs = self.valid_costs[i - 2 * Neval:i]

            # move parameters
            self.update()

            # perform monitoring
            if i % check_epoch == 0:
                self.model.save(train_costs, valid_costs, self.velocities)
                if verbose:
                    print '#' * 80 + '\n' * 1
                    m = 'Epoch:5, Train cost: %0.3e, Valid cost: %0.3e'
                    m = m % (i, train_costs, valid_costs)
                    print m + '\n' * 1 '#' * 80 + '\n' * 3
                if i > 2 * Neval:
                    done = self.check_if_done(block_of_costs)
                    if done:
                        break

    def check_if_done(self, block_of_costs):
        """
        Check if model is done, if so save params
        """
        prev = np.median(block_of_costs[:self.Neval])
        curr = np.median(block_of_costs[self.Neval:])
        return ((prev - curr) / prev) < self.tol
