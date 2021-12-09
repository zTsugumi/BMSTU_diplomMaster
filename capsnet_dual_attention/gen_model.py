import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D,\
    Dense, Reshape, LeakyReLU, Conv2DTranspose, BatchNormalization
from .bottleneck import BottleNeck
from .part_capsule import PartCapsule
from utils import pose_to_affine, affine_inverse,\
    transform_affine, transform_occlusion

params_GEO = {
    'name': 'GEO',
    'n_caps': 11,   # number of capsules
    'd_caps': 8     # pose dimension
}


def encoder_graph(params, input_shape):
  # Input
  # -> (None, 128, 128, 3)
  input = keras.Input(input_shape)

  # Base conv
  # -> (None, 32, 32, 64)
  features = keras.Sequential(
      name='base_conv',
      layers=[
          Conv2D(
              filters=32, kernel_size=3,
              strides=1, padding='same',
              use_bias=False, activation='relu'),
          AvgPool2D(pool_size=(2, 2)),  # Try to change it to MaxPool2D
          Conv2D(
              filters=32, kernel_size=3,
              strides=1, padding='same',
              use_bias=False, activation='relu'),
          AvgPool2D(pool_size=(2, 2)),
          Conv2D(
              filters=64, kernel_size=3,
              strides=1, padding='same',
              use_bias=False, activation='relu'),
          AvgPool2D(pool_size=(2, 2)),
          Conv2D(
              filters=128, kernel_size=1,
              strides=1, padding='same',
              use_bias=False, activation='relu'),
      ])(input)

  # Bottle neck layer
  # -> (None, 32, 32, 128)
  #features = BottleNeck(n_block=2, block_base=64, max_features=128)(features)

  # Capsule formulation
  # -> pose:      (None, n_caps, d_caps)
  # -> rotate:    (None, n_caps, 2)
  # -> shift:     (None, n_caps, 2)
  # -> scale:     (None, n_caps, 2)
  # -> occlusion: (None, n_caps)
  pose, rotate, shift, scale, occlusion = PartCapsule(
      params['n_caps'], params['d_caps'])(features)

  return keras.Model(
      inputs=input,
      outputs=[pose, rotate, shift, scale, occlusion],
      name='Encoder'
  )


def decoder_graph(params):
  # Input
  # -> (None, n_caps, d_caps)
  input = keras.Input((params['n_caps'], params['d_caps']))

  # Recontructed image
  # -> (None, 128, 128, 3)
  img_outs = []
  base_conv = [keras.Sequential(
      layers=[
          Dense(4096, activation='relu'),
          Reshape((16, 16, 16)),
          BatchNormalization(momentum=0.8),
          Conv2DTranspose(
              filters=64, kernel_size=3,
              strides=1, padding='same',
              activation=LeakyReLU(0.2)),
          Conv2DTranspose(
              filters=32, kernel_size=3,
              strides=2, padding='same',
              activation=LeakyReLU(0.2)),
          Conv2DTranspose(
              filters=16, kernel_size=3,
              strides=2, padding='same',
              activation=LeakyReLU(0.2)),
          Conv2DTranspose(
              filters=8, kernel_size=3,
              strides=2, padding='same',
              activation=LeakyReLU(0.2)),
          Conv2DTranspose(
              filters=3, kernel_size=3,
              strides=1, padding='same',
              activation=LeakyReLU(0.2)),
      ]) for _ in range(params['n_caps'])]

  for i in range(params['n_caps']):
    img_reconstructed = base_conv[i](input[:, i])
    img_outs.append(img_reconstructed)

  # -> (None, n_caps, 128, 128, 3)
  img_outs = tf.stack(img_outs, axis=1)

  return keras.Model(
      inputs=input,
      outputs=img_outs,
      name='Decoder'
  )


def build_graph(name, input_shape, mode, verbose=False):
  '''
  This contructs the whole architecture of Capsnet dual attention
  (Encoder + Decoder)
  '''
  # Setup params
  if name == 'GEO':
    params = params_GEO
  else:
    raise RuntimeError(f'name {name} not recognized')

  input1 = keras.Input(input_shape)                                 # None, 2, 128, 128, 3
#   input1, input2 = tf.unstack(inputs, axis=1)                       # None, 128, 128, 3
#   input_shape = input_shape[1:]

  # Encoder
  encoder = encoder_graph(params, input_shape)
  # Decoder
  decoder = decoder_graph(params)

  if verbose:
    encoder.summary()
    decoder.summary()

  # Forward
  pose1, rotate1, shift1, scale1, occlusion1 = encoder(input1)
  # pose2, rotate2, shift2, scale2, occlusion2 = encoder(input2)

  affine1 = pose_to_affine((rotate1, shift1, scale1))               # None, n_caps, 2, 3
  affine_inv1 = affine_inverse(affine1)
  # affine2 = pose_to_affine((rotate2, shift2, scale2))
  # affine_inv2 = affine_inverse(affine2)

  # Forward
  imgs1 = decoder(pose1)                                            # None, n_caps, 128, 128, 3
  # imgs2 = decoder(pose2)

  h, w, c = imgs1.shape[2:]
  imgs1 = tf.reshape(imgs1, (-1, h, w, c))                          # None*n_caps, 128, 128, 3
  affine_inv1 = tf.reshape(affine_inv1, (-1, 2, 3))                 # None*n_caps, 2, 3

  transformed_imgs1 = transform_affine(imgs1, affine_inv1, (h, w))  # None*n_caps, 128, 128, 3
  # transformed_imgs2 = transform_affine(imgs2, affine2, (h, w))

  transformed_imgs1 = tf.reshape(transformed_imgs1, (-1, params['n_caps'], h, w, c))

  # transformed_imgs1 = tl.layers.transformer(imgs1, affine1, (h, w))

  final_imgs1 = transform_occlusion(transformed_imgs1, occlusion1)  # None, 128*128*3
  # final_imgs2 = transform_occlusion(transformed_imgs2, occlusion2)

  # final_imgs = tf.stack([final_imgs1, final_imgs2], axis=1)

  if mode == 'train':
    return keras.Model(
        inputs=input1,
        outputs=final_imgs1,
        name='CapsNetDualAttention'
    )
  elif mode == 'test':
    return keras.Model(
        inputs=input1,
        outputs=[final_imgs1, input1, transformed_imgs1, imgs1],
        name='CapsNetDualAttention'
    )
  else:
    raise RuntimeError(f'mode {mode} not recognized')
