import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from capsnet.layers import PrimaryCaps, DigitCaps, Length, Mask

params_CIFAR10 = {
    'conv_filters': 256,
    'conv_kernel': 9,
    'conv_stride': 1,

    'caps_primary': 64,
    'caps_primary_dim': 8,
    'caps_primary_kernel': 9,
    'caps_primary_stride': 2,

    'caps_digit_dim': 16
}


def encoder_graph(params, input_shape, output_class, r):
  '''
  This constructs the Encoder layers of Capsule Network
  '''
  inputs = keras.Input(shape=input_shape)

  x = keras.layers.Conv2D(
      filters=params['conv_filters'],
      kernel_size=params['conv_kernel'],
      strides=params['conv_stride'],
      padding='valid',
      activation='relu')(inputs)
  primary_caps = PrimaryCaps(
      C=params['caps_primary'],
      L=params['caps_primary_dim'],
      k=params['caps_primary_kernel'],
      s=params['caps_primary_stride'])(x)
  digit_caps = DigitCaps(
      C=output_class,
      L=params['caps_digit_dim'],
      r=r)(primary_caps)
  digit_caps_len = Length()(digit_caps)

  return keras.Model(
      inputs=inputs,
      outputs=[primary_caps, digit_caps, digit_caps_len],
      name='Encoder'
  )


def decoder_graph(params, input_shape, output_class):
  '''
  This constructs the Decoder layers of Capsule Network
  '''
  inputs = keras.Input(
      output_class*params['caps_digit_dim'])

  x = keras.layers.Dense(512, activation='relu')(inputs)
  x = keras.layers.Dense(1024, activation='relu')(x)
  x = keras.layers.Dense(np.prod(input_shape), activation='sigmoid')(x)
  x = keras.layers.Reshape(input_shape)(x)

  return keras.Model(
      inputs=inputs,
      outputs=x,
      name='Decoder'
  )


def build_graph(name, input_shape, output_class, mode, r, verbose=False):
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

  encoder = encoder_graph(params, input_shape, output_class, r)
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

  decoder = decoder_graph(params, input_shape, output_class)
  x_reconstruct = decoder(masked)

  if verbose:
    decoder.summary()

  if mode == 'train':
    return keras.Model(
        inputs=[inputs, y_true],
        outputs=[digit_caps_len, x_reconstruct],
        name='CapsNet'
    )
  elif mode == 'test':
    return keras.Model(
        inputs=[inputs],
        outputs=[digit_caps_len, x_reconstruct],
        name='CapsNet'
    )
  elif mode == 'exp':
    return keras.Model(
        inputs=[inputs, noise],
        outputs=[digit_caps_len, x_reconstruct],
        name='CapsNet'
    )
  else:
    raise RuntimeError(f'mode {mode} not recognized')
