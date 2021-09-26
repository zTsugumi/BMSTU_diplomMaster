import numpy as np
import tensorflow as tf


def checkpoint(dir_model, dir_log):
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        dir_model + '/weights-{epoch:02d}.h5',
        monitor='val_Encoder_accuracy',
        save_weights_only=True,
        verbose=1
    )

    board = tf.keras.callbacks.TensorBoard(
        dir_log + '/log',
        histogram_freq=0
    )

    return model_checkpoint, board


def lr_sched(lr, lr_decay):
    return tf.keras.callbacks.LearningRateScheduler(
        schedule=lr_func(lr, lr_decay))


def lr_func(lr, lr_decay):
    def fn(epoch):
        lr_new = lr * (lr_decay ** epoch)
        return lr_new if lr_new >= 5e-5 else 5e-5
    return fn
