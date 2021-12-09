import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, Flatten
from utils import MLP


class PartCapsule(keras.layers.Layer):
  '''Transform features to capsules and transformation params
  Similar to Spatial Transformer
  '''

  def __init__(self, n_caps, d_caps, **kwargs):
    super(PartCapsule, self).__init__(**kwargs)
    self.n_caps = n_caps
    self.d_caps = d_caps

    self.volume = keras.Sequential(
        name='Pose',
        layers=[
            Conv2D(
                filters=n_caps*d_caps*4, kernel_size=4,
                strides=2, padding='same',
                activation=LeakyReLU(0.2)
            ),
            Conv2D(
                filters=n_caps*d_caps*2, kernel_size=4,
                strides=2, padding='same',
                activation=LeakyReLU(0.2)
            ),
            Conv2D(                                                               # None, 4, 4, n_caps*d_caps
                filters=n_caps*d_caps, kernel_size=7,
                strides=2, padding='same',
                activation=LeakyReLU(0.2)
            ),
            # Flatten(),
            # Dense(                                                                # None, n_caps*d_caps
            #     n_caps*d_caps,
            #     activation='tanh'
            # )
        ]
    )

    # self.pose = [MLP([d_caps], activation=None)
    #              for _ in range(n_caps)]
    # self.rotate = [MLP([d_caps // 2, 2], activation=None)
    #                for _ in range(n_caps - 1)]
    # self.shift = [MLP([d_caps // 2, 2], activation=None)
    #               for _ in range(n_caps - 1)]
    # self.scale = [MLP([d_caps // 2, 2], activation=None)
    #               for _ in range(n_caps - 1)]
    # self.occlusion = [MLP([d_caps // 2, 1], activation=None)
    #                   for _ in range(n_caps - 1)]

    self.pose = Dense(n_caps*d_caps, activation=None, name='Pose')
    self.rotate = Dense((n_caps - 1) * 2, activation='tanh', name='Rotate')     # sin, cos of rotation
    self.shift = Dense((n_caps - 1) * 2, activation='tanh', name='Shift')       # tx, ty
    self.scale = Dense((n_caps - 1) * 2, activation='tanh', name='Scale')       # sx, sy
    self.occlusion = Dense(n_caps - 1, activation='tanh', name='Occlusion')     # occlusion param

  def get_config(self):
    config = {
        'n_caps': self.n_caps,
        'd_caps': self.d_caps,
        'volume': self.volume,
        'pose': self.pose,
        'rotate': self.rotate,
        'shift': self.shift,
        'scale': self.scale,
        'occlusion': self.occlusion
    }
    base_config = super(PartCapsule, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, input):                                          # None, 32, 32, 128
    b = tf.shape(input)[0]
    features = self.volume(input)                                 # None, 4, 4, n_caps*d_caps
    features = Flatten()(features)

    # Pose
    pose = self.pose(features)
    pose = tf.reshape(pose, (-1, self.n_caps, self.d_caps))       # None, n_caps, d_caps
    pose_caps, pose_bg = tf.split(
        pose, [self.n_caps - 1, 1], axis=1)                       # None, n_caps - 1, d_caps
    pose_caps_split = tf.unstack(pose_caps, axis=1)               # [None, d_caps]

    # Rotate
    rotate_zeros = tf.constant(
        [[[1, 0]]],
        dtype=tf.float32)
    rotate_zeros = tf.repeat(rotate_zeros, b, axis=0)             # None, 1, 2
    rotate = self.rotate(features)
    rotate = tf.reshape(rotate, (-1, self.n_caps - 1, 2))
    # rotate = [self.rotate[i](pose_caps_split[i])
    #           for i in range(self.n_caps - 1)]
    # rotate = tf.stack(rotate, axis=1)                             # None, n_caps - 1, 2
    rotate = tf.concat([rotate, rotate_zeros], axis=1)            # None, n_caps, 2
    rotate = rotate / (tf.norm(rotate, ord=2, axis=-1, keepdims=True) + 1e-10)

    # Shift
    shift_zeros = tf.zeros_like(
        rotate_zeros,
        dtype=rotate_zeros.dtype)                                 # None, 1, 2
    shift = self.shift(features)
    shift = tf.reshape(shift, (-1, self.n_caps - 1, 2))
    # shift = [self.shift[i](pose_caps_split[i])
    #          for i in range(self.n_caps - 1)]
    # shift = tf.stack(shift, axis=1)                               # None, n_caps - 1, 2
    shift = tf.concat([shift, shift_zeros], axis=1)               # None, n_caps, 2

    # Scale
    scale_zeros = tf.ones_like(
        rotate_zeros,
        dtype=rotate_zeros.dtype)                                 # None, 1, 2
    scale = self.scale(features)
    scale = tf.reshape(scale, (-1, self.n_caps - 1, 2))
    # scale = [self.scale[i](pose_caps_split[i])
    #          for i in range(self.n_caps - 1)]
    # scale = tf.stack(scale, axis=1)                               # None, n_caps - 1, 2
    scale = tf.tanh(scale) * 0.9 + 1
    scale = 1 / scale
    scale = tf.concat([scale, scale_zeros], axis=1)               # None, n_caps, 2

    # Occlusion
    occlusion_zeros = tf.constant(
        [[0]],
        dtype=rotate_zeros.dtype)
    occlusion_zeros = tf.repeat(occlusion_zeros, b, axis=0)       # None, 1
    occlusion = self.occlusion(features)
    # occlusion = [self.occlusion[i](pose_caps_split[i])
    #              for i in range(self.n_caps - 1)]
    # occlusion = tf.stack(occlusion, axis=1)                       # None, n_caps - 1, 1
    # occlusion = tf.squeeze(occlusion, axis=-1)                    # None, n_caps - 1
    occlusion = tf.concat([occlusion, occlusion_zeros], axis=1)   # None, n_caps

    pose_caps = tf.stack(pose_caps_split, axis=1)                 # None, n_caps - 1, d_caps
    pose = tf.concat([pose_caps, pose_bg], axis=1)                # None, n_caps, d_caps

    return (pose, rotate, shift, scale, occlusion)
