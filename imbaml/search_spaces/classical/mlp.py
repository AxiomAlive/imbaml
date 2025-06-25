from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.neural_network import MLPClassifier

from imbaml.search_spaces import EstimatorSpaceGenerator


class MLPClassifierGenerator(EstimatorSpaceGenerator):
    activation = hp.pchoice('mlp.activation', [(0.2, "identity"), (0.2, "logistic"), (0.2, "tanh"), (0.4, "relu")])
    solver = hp.pchoice('mlp.solver', [(0.2, "lbfgs"), (0.2, "sgd"), (0.6, "adam")])
    alpha = hp.uniform('mlp.alpha', 1e-4, 0.01)
    learning_rate = hp.choice('mlp.learning_rate', ["constant", "invscaling", "adaptive"])
    learning_rate_init = hp.uniform('mlp.learning_rate_init', 1e-4, 0.1)
    power_t = hp.uniform('mlp.power_t', 0.1, 0.9)
    max_iter = scope.int(hp.uniform('mlp.max_iter', 150, 350))
    tol = hp.uniform('mlp.tol', 1e-4, 0.01)
    momentum = hp.uniform('mlp.momentum', 0.8, 1.0)
    nesterovs_momentum = hp.choice('mlp.nesterovs_momentum', [True, False])
    early_stopping = hp.choice('mlp.early_stopping', [True, False])
    validation_fraction = hp.uniform('mlp.validation_fraction', 0.01, 0.2)
    beta_1 = hp.uniform('mlp.beta_1', 0.8, 1.0)
    beta_2 = hp.uniform('mlp.beta_2', 0.95, 1.0)
    epsilon = hp.uniform('mlp.epsilon', 1e-9, 1e-5)
    n_iter_no_change = hp.choice('mlp.n_iter_no_change', [10, 20, 30])
    max_fun = scope.int(hp.uniform('mlp.max_fun', 1e4, 3e4))

    @classmethod
    def generate(cls, model_class=None):
        param_map = super().generate()
        param_map.update({'model_class': MLPClassifier})

        return param_map


