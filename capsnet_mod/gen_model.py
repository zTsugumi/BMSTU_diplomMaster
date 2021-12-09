import tensorflow.keras as keras
from capsnet_mod.layers import PrimaryCaps, DigitCaps, Length, Mask

params_CIFAR10 = {
    'name': 'CIFAR10',
    'conv_kernel': 9,
    'conv_stride': 1,

    'caps_primary': 64,
    'caps_primary_dim': 8,
    'caps_primary_kernel': 9,
    'caps_primary_stride': 2,

    'caps_digit_dim': 16
}


def encoder_graph(params, input_shape, output_class):
  '''
  This constructs the Encoder layers of Modified Capsule Network
  '''
  inputs = keras.Input(input_shape)

  if params == params_CIFAR10:
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal',
        activation=None)(inputs)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal',
        activation=None)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal',
        activation=None)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal',
        activation=None)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(
        filters=512,
        kernel_size=3,
        strides=2,
        padding='valid',
        kernel_initializer='he_normal',
        activation=None)(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.BatchNormalization()(x)
  else:
    raise RuntimeError('model not recognized')

  primary_caps = PrimaryCaps(
      C=params['caps_primary'],
      L=params['caps_primary_dim'],
      k=params['caps_primary_kernel'],
      s=params['caps_primary_stride'])(x)
  digit_caps = DigitCaps(
      C=output_class,
      L=params['caps_digit_dim'])(primary_caps)
  digit_caps_len = Length()(digit_caps)

  return keras.Model(
      inputs=inputs,
      outputs=[primary_caps, digit_caps, digit_caps_len],
      name='Encoder'
  )


def decoder_graph(params):
  '''
  This constructs the Decoder layers
  '''
  inputs = keras.Input(
      params['caps_digit_dim']
  )

  if params == params_CIFAR10:
    x = keras.layers.Dense(1024, activation='relu')(inputs)
    x = keras.layers.Reshape((8, 8, 16))(x)
  else:
    raise RuntimeError(f'model not recognized')

  x = keras.layers.BatchNormalization(momentum=0.8)(x)
  x = keras.layers.Conv2DTranspose(64, 3, (1, 1), padding='same')(x)
  x = keras.layers.Conv2DTranspose(32, 3, (2, 2), padding='same')(x)
  x = keras.layers.Conv2DTranspose(16, 3, (2, 2), padding='same')(x)
  x = keras.layers.Conv2DTranspose(8, 3, (1, 1), padding='same')(x)

  if params == params_CIFAR10:
    x = keras.layers.Conv2DTranspose(3, 3, (1, 1), padding='same')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Reshape((32, 32, 3))(x)
  else:
    raise RuntimeError(f'model not recognized')

  return keras.Model(
      inputs=inputs,
      outputs=x,
      name='Decoder'
  )


def build_graph(name, input_shape, output_class, mode, verbose=False):
  '''
  This contructs the whole architecture of Capsule Network 
  (Encoder + Decoder)
  '''
  # Setup params
  if name == 'CIFAR10':
    params = params_CIFAR10
  else:
    raise RuntimeError(f'name {name} not recognized')

  # Encoder
  inputs = keras.Input(input_shape)
  y_true = keras.Input(output_class)

  encoder = encoder_graph(params, input_shape, output_class)

  primary_caps, digit_caps, digit_caps_len = encoder(inputs)

  if verbose:
    encoder.summary()

  # Decoder
  if mode == 'train':
    masked = Mask()([digit_caps, y_true])
  elif mode == 'test':
    masked = Mask()(digit_caps)
  elif mode == 'exp':
    noise = keras.Input(
        (output_class, params['caps_digit_dim']))
    digit_caps_noise = keras.layers.add([digit_caps, noise])
    masked = Mask()(digit_caps_noise)
  else:
    raise RuntimeError(f'mode {mode} not recognized')

  decoder = decoder_graph(params)

  x_reconstruct = decoder(masked)

  if verbose:
    decoder.summary()

  if mode == 'train':
    return keras.Model(
        inputs=[inputs, y_true],
        outputs=[digit_caps_len, x_reconstruct],
        name='CapsNetMod'
    )
  elif mode == 'test':
    return keras.Model(
        inputs=[inputs],
        outputs=[digit_caps_len, x_reconstruct],
        name='CapsNetMod'
    )
  elif mode == 'exp':
    return keras.Model(
        inputs=[inputs, noise],
        outputs=[digit_caps_len, x_reconstruct],
        name='CapsNetMod'
    )
  else:
    raise RuntimeError(f'mode {mode} not recognized')
