import numpy as np
import pandas as pd
from abc import abstractmethod
from datetime import datetime
from sys import stdout

from sklearn.model_selection import KFold, StratifiedKFold

from pySOT import SyncStrategyNoConstraints, LatinHypercube, RBFInterpolant, CandidateDYCORS, CubicKernel, LinearTail
from poap.controller import SerialController

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


class BaseNeuralNetwork(object):
    """ A base class for Neural Network predictors. """

    def __init__(self):
        self.best_loss = np.inf
        self.best_params = None
        self.hyperparams = list(self.hyper_ranges.keys())
        self.dim = len(self.hyperparams)
        self.hyper_map = {self.hyperparams[i]: i for i in range(self.dim)}
        self.xlow = np.array([self.hyper_ranges[key][0] for key in self.hyperparams])
        self.xup = np.array([self.hyper_ranges[key][1] for key in self.hyperparams])

    @abstractmethod
    def initialize_model(self, params):
        """ Create and return a compiled Keras model. """

    def fit(self, params, stratified_batches=False):
        print("Creating model with the following parameters:")
        print(params)
        self.model = self.initialize_model(params)

        print("Begin Training.")
        param_list = [params[self.hyperparams[i]] for i in range(len(self.hyperparams))]
        self.model = self.train(self.model, self.x, self.y, param_list, stratified_batches)

        self.fitted = True

    def evaluate(self, x, y):
        assert self.fitted, "First train the model using fit()"
        return self.model.evaluate(x, y)

    def predict(self, x):
        assert self.fitted, "First train the model using fit()"
        predictions = self.model.predict(x)
        return predictions

    def get_splits(self, n_folds):
        """ Get k-fold splits of the data with or without stratification. """
        if self.stratify_labels is None:
            return KFold(n_splits=n_folds).split(self.x)
        else:
            return StratifiedKFold(n_splits=n_folds).split(self.x, self.stratify_labels)

    def get_stratified_batch(self, x, y, batch_size, num_nonzero=5):
        """ Get a batch of size batch_size with a fixed number of samples with a nonzero target. """
        nonzeros = np.nonzero(y)[0]
        nonzero_index = nonzeros[np.random.randint(low=0, high=len(nonzeros), size=num_nonzero)]
        # we don't really care if there would be an extra nonzero in the batch
        zero_index = np.random.randint(low=0, high=len(y), size=batch_size-num_nonzero)
        x_batch = np.concatenate([x[nonzero_index], x[zero_index]], axis=0)
        y_batch = np.concatenate([y[nonzero_index], y[zero_index]], axis=0)
        return x_batch, y_batch

    def train_with_stratified_batches(self, model, x, y, params, num_nonzero=5):
        """ Train with stratified batches (nonzero targets in every batch). """
        print("Training on stratified batches with {} nonzero targets per batch of {}"
              .format(num_nonzero, params[self.hyper_map['batch_size']]))

        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        num_batches = params[self.hyper_map['epochs']] * len(x) // params[self.hyper_map['batch_size']]

        for i in range(num_batches):
            stdout.write("\rBatch: {}/{}".format(i, num_batches))
            stdout.flush()
            x_batch, y_batch = self.get_stratified_batch(x, y, params[self.hyper_map['batch_size']], num_nonzero=num_nonzero)
            model.train_on_batch(x_batch, y_batch, sample_weight=None, class_weight=None)

        print("Done training.")
        return model

    def train(self, model, x, y, params, stratified_batches=True):
        """ Wrapper to decide whether to train on stratified batches or not. """
        if stratified_batches:
            return self.train_with_stratified_batches(model, x, y, params)
        else:
            model.fit(x, y, epochs=int(params[self.hyper_map['epochs']]),
                      batch_size=int(params[self.hyper_map['batch_size']]))
            return model

    def cross_validate(self, params, k):
        """ Cross validation with stratification. """
        print("Cross validating in {} folds..".format(self.nfolds))
        scores = []

        for train_index, val_index in self.get_splits(k):

            train_x = self.x[train_index]
            train_y = self.y[train_index]
            val_x = self.x[val_index]
            val_y = self.y[val_index]

            # initialize keras model
            model = self._initialize_model_with_param_list(params)

            # train
            model = self.train(model, train_x, train_y, params)

            # test
            score = model.evaluate(val_x, val_y)
            scores.append(score)

        return scores

    def objfunction(self, params):
        """ The overall objective function to provide to pySOT's black box optimization.

        :param params: The parameters to use for the function evaluation (array like).
        :returns: Mean loss on the validation set.
        """

        self.exp_number += 1
        print("-------------\nExperiment {}.\n-------------".format(self.exp_number))
        for p in self.hyperparams:
            print(p+": "+str(params[self.hyper_map[p]]))
        print("-------------")

        # run the experiment nfolds times and print mean and std.
        scores = self.cross_validate(params, self.nfolds)
        print("Scores: {}.\nMean: {}. Standard deviation: {}%".format(scores, np.mean(scores), np.std(scores)))

        # log after every function call / for every set of parameters
        if self.log:
            # log to object and to files
            self.param_log = pd.concat([self.param_log, pd.DataFrame(np.reshape(params, (1, self.dim)), columns=self.hyperparams)])
            self.scores_log = pd.concat([self.scores_log, pd.DataFrame(np.reshape(scores, (1, self.nfolds)), columns=np.arange(1, self.nfolds + 1))])
            self.param_log.to_csv(self.log_path + self.name + "_params_log.csv", index=False)
            self.scores_log.to_csv(self.log_path + self.name + "_scores_log.csv", index=False)

        # prevent memory buildup
        # K.clear_session()

        # keep track of best scores and params for convenience
        if np.mean(scores) < self.best_loss:
            self.best_loss = np.mean(scores)
            self.best_params = {self.hyperparams[i]: params[i] for i in range(self.dim)}

        # return mean value for pySOT to minimize
        return np.mean(scores)

    def tune_with_HORD(self, max_evaluations, log=True, log_path="./"):
        """ Automatically tune hyperparameters using HORD (Ilievski et al., 2017).

        :param max_evaluations: maximum function evaluations (so maximum number of parameter settings to try).
        """
        if log:
            self.log = True

        self.exp_number = 0
        self.param_log = pd.DataFrame(columns=self.hyperparams)
        self.scores_log = pd.DataFrame(columns=np.arange(1, self.nfolds + 1))
        self.log_path = log_path

        # create controller
        controller = SerialController(self.objfunction)
        # experiment design
        exp_des = LatinHypercube(dim=self.dim, npts=2*self.dim+1)
        # use a cubic RBF interpolant with a linear tail
        surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=max_evaluations)
        # use DYCORS with 100d candidate points
        adapt_samp = CandidateDYCORS(data=self, numcand=100*self.dim)
        # strategy
        strategy = SyncStrategyNoConstraints(worker_id=0, data=self, maxeval=max_evaluations, nsamples=1,
                                             exp_design=exp_des, response_surface=surrogate,
                                             sampling_method=adapt_samp)
        controller.strategy = strategy

        # Run the optimization strategy
        start_time = datetime.now()
        result = controller.run()

        print('Best value found: {0}'.format(result.value))
        print('Best solution found: {0}\n'.format(
            np.array_str(result.params[0], max_line_width=np.inf,
                         precision=5, suppress_small=True)))

        print('Started: '+str(start_time)+'. Ended: ' + str(datetime.now()))


class MultiLayerPerceptron(BaseNeuralNetwork):
    """ A Dense Neural Network predictor with built-in tuning procedure.

    Parameters
    ----------
    x: np.array
        The training features.
    y: np.array
        The training labels.
    loss: str
        The loss function, must be accapted by Keras.
    prediction_type: str, one of {'regression', 'classification'}
        Whether you want a regressor or classifier.
    stratify_labels: array-like, optional
        The labels of classes to use in stratification when cross validating. Default is None,
        in which casee no stratification is performed.
    """
    name = 'Multi-Layer Perceptron'

    def __init__(self, x, y, loss='mean_squared_error', nfolds=10, prediction_type="regression",
                 stratify_labels=None):

        # specify type of prediction
        self.prediction_type = prediction_type

        # data
        self.x = x
        self.y = y
        self.stratify_labels = stratify_labels

        # optimization parameters
        self.loss = loss
        self.nfolds = nfolds

        # specify the parameter ranges for tuning as [min, max].
        # first continuous, then integer params.
        self.activation_candidates = ['tanh', 'relu', 'selu']
        self.optimizer_candidates = [keras.optimizers.RMSprop(), keras.optimizers.Adam()]

        self.hyper_ranges = {'learning_rate': [0.0001, 0.01],
                             'lr_decay': [0.000000001, 0.00001],
                             'dropout_rate': [0.0, 0.7],
                             'neurons': [16, 512],
                             'hidden_layers': [2, 6],
                             'batch_size': [16, 512],
                             'epochs': [3, 50],
                             'activation': [0, len(self.activation_candidates)-1],
                             'optimizer': [0, len(self.optimizer_candidates)-1]
                             }

        super(MultiLayerPerceptron, self).__init__()
        # specify the continuous and the integer parameters
        self.continuous = np.arange(0, 3)
        self.integer = np.arange(3, self.dim)

    def initialize_model(self, params):
        try:
            flat_params = [params[name] for name in self.hyperparams]
        except KeyError:
            raise ValueError("You must specify all hyperparameters.")

        return self._initialize_model_with_param_list(flat_params)

    def _initialize_model_with_param_list(self, params):
        """ Create a multi layer perceptron in Keras.

        Parameters
        ----------
        params: array-like
            Hyperparameters that define the model. In the order of `hyperparams`.

        Returns
        -------
        model: Keras.model object
            Compiled Keras model of a dense neural network.
        """
        m = self.hyper_map
        act = self.activation_candidates[int(params[m['activation']])]

        model = Sequential()
        for layer in range(int(params[m['hidden_layers']])):
            if layer == 0:
                model.add(Dense(int(params[m['neurons']]), activation=act, input_shape=(self.x.shape[1],)))
            model.add(Dense(int(params[m['neurons']]), activation=act))

        model.add(Dropout(params[m['dropout_rate']]))

        if self.prediction_type == "classification":
            model.add(Dense(self.y.shape[1], activation='sigmoid'))
        elif self.prediction_type == "regression":
            model.add(Dense(1, activation="relu"))
        else:
            raise ValueError("prediction_type must be one of ['regression', 'prediction']. Received {}"
                             .format(self.prediction_type))

        optimizer = self.optimizer_candidates[int(params[m['optimizer']])]
        optimizer.lr = params[m['learning_rate']]
        optimizer.decay = params[m['lr_decay']]

        model.compile(optimizer=optimizer, loss=self.loss)

        return model
