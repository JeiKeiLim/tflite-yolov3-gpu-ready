import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class YoloLRScheduler(Callback):
    def __init__(self, initial_lr=0.001, last_lr=0.0001, start_step_epoch=0, batch_size=32, steps_per_epoch=100, n_data=10000):
        super().__init__()
        self.initial_lr = initial_lr
        self.last_lr = last_lr
        self.step_epoch = start_step_epoch
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.n_data = n_data
        self.n_seen = self.batch_size * self.step_epoch * self.steps_per_epoch
        self.epoch_ratio = (self.n_seen / self.n_data) % 1

    def batch_update(self, n):
        self.n_seen = (self.n_seen + n) % self.n_data
        self.epoch_ratio = (self.n_seen / self.n_data) % 1

        lr = (self.initial_lr*(1-self.epoch_ratio)) + (self.last_lr*self.epoch_ratio)
        tf.keras.backend.set_value(self.model.optimizer.lr, tf.keras.backend.get_value(lr))

    def on_batch_end(self, batch, logs=None):
        self.batch_update(self.batch_size)


class UnfreezeScheduler(Callback):
    def __init__(self, epochs=0, wait_epoch=10, unfreeze_step=1, unfreeze_from=229, enable=True):
        super().__init__()
        self.epochs = epochs
        self.wait_epoch = wait_epoch
        self.unfreeze_step = unfreeze_step
        self.unfreeze_from = unfreeze_from
        self.enable = enable

    def on_epoch_end(self, epoch, logs=None):
        self.epochs += 1

        n_unfreeze = (self.epochs-self.wait_epoch) * self.unfreeze_step
        unfreeze_to = max(self.unfreeze_from-n_unfreeze, 0)

        if unfreeze_to < self.unfreeze_from:
            print("... Unfreezing layer {:03d}~{:03d}".format(unfreeze_to, self.unfreeze_from))

        for i in range(self.unfreeze_from, unfreeze_to, -1):
            self.model.layers[i].trainable = True

