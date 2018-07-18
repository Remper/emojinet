import logging
import numpy as np
from keras.callbacks import Callback
from sklearn import metrics


class EvalCallback(Callback):
    def __init__(self, name, X_test, Y_test, period=1):
        super(EvalCallback, self).__init__()

        self.name = name
        self.X_test = X_test
        self.Y_test = Y_test
        self.period = period

    def get_predictions(self) -> list:
        return [np.argmax(prediction) for prediction in self.model.predict(self.X_test)]

    def get_f1(self, predictions: list) -> float:
        return metrics.f1_score(self.Y_test, predictions, average="macro")

    def evaluate(self) -> None:
        Y_test_pred = self.get_predictions()
        name = self.name
        if len(name) > 10:
            name = name[:8] + ".."
        logging.info("[%10s] Accuracy: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f" % (
            name,
            metrics.accuracy_score(self.Y_test, Y_test_pred),
            metrics.precision_score(self.Y_test, Y_test_pred, average="macro"),
            metrics.recall_score(self.Y_test, Y_test_pred, average="macro"),
            self.get_f1(Y_test_pred)
        ))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch+1) % self.period == 0:
            self.evaluate()


class ValidationEarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
    """

    def __init__(self,
                 monitor: EvalCallback,
                 min_delta=0,
                 patience=1):
        super(ValidationEarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.best = - np.Inf

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.monitor.get_f1(self.monitor.get_predictions())
        if np.greater(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logging.info('Epoch %d: early stopping' % (self.stopped_epoch + 1))