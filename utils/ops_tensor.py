import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Dense


class Conv2DDown(keras.layers.Layer):
  def __init__(self, out_features, kernel_size=3, strides=2):
    super(Conv2DDown, self).__init__()
    self.down_block = keras.Sequential(
        name='down_block',
        layers=[
            Conv2D(
                filters=out_features,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                activation=None),
            BatchNormalization(),
            Activation('relu')
        ])

  def get_config(self):
    config = {
        'down_block': self.down_block
    }
    base_config = super(Conv2DDown, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    # outputs = BatchNormalization()(outputs)
    # outputs = Activation('relu')(outputs)
    # outputs = keras.layers.AvgPool2D(pool_size=(2, 2))(outputs)
    outputs = self.down_block(inputs)
    return outputs


class Conv2DUp(keras.layers.Layer):
  def __init__(self, out_features, kernel_size=3, strides=2):
    super(Conv2DUp, self).__init__()
    self.up_block = keras.Sequential(
        name='up_block',
        layers=[
            Conv2DTranspose(
                filters=out_features,
                kernel_size=kernel_size,
                strides=strides,
                padding='same',
                activation=None),
            BatchNormalization(),
            Activation('relu')
        ])

  def get_config(self):
    config = {
        'up_block': self.up_block
    }
    base_config = super(Conv2DUp, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    # outputs = self.conv2dtranspose(inputs)
    # outputs = BatchNormalization()(outputs)
    # outputs = Activation('relu')(outputs)
    outputs = self.up_block(inputs)
    return outputs


class MLP(keras.layers.Layer):
  def __init__(self, outs, activation, **kwargs):
    super(MLP, self).__init__(**kwargs)

    self.layers = keras.Sequential()

    for out in outs:
      self.layers.add(Dense(out, activation=activation))

  def get_config(self):
    config = {
        'layers': self.layers,
    }
    base_config = super(MLP, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, input):
    return self.layers(input)


class CoAttention(keras.layers.Layer):
  def __init__(self, **kwargs):
    super(CoAttention, self).__init__(**kwargs)
  
  def get_config(self):
    config = {
        # 'layers': self.layers,
    }
    base_config = super(MLP, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    input1, input2 = inputs

    