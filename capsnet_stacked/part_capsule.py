import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import sonnet as snt
from capsnet_stacked.math_ops import geo_transform


def ImageEncoder(params, input_shape):
  '''Convert image into part capsules
  '''
  n_caps = params['n_caps']
  n_caps_dims = params['n_caps_dims']
  n_features = params['n_features']
  noise_scale = params['noise_scale']

  splits = [n_caps_dims, n_features, 1]
  n_dims = sum(splits)

  img = tf.keras.Input(shape=input_shape)
  batch_size = tf.shape(img)[0]

  cnn_encoder = keras.Sequential([
      keras.layers.Conv2D(128, 3, 2, 'valid', activation='relu'),
      keras.layers.Conv2D(128, 3, 2, 'valid', activation='relu'),
      keras.layers.Conv2D(128, 3, 1, 'valid', activation='relu'),
      keras.layers.Conv2D(128, 3, 1, 'valid', activation='relu'),
  ])

  img_embedding = cnn_encoder(img)                                    # (32, 2, 2, 128)

  # Conv Attention
  h = keras.layers.Conv2D(n_dims * n_caps + n_caps, 1, 1)(
      img_embedding)                                                  # (32, 2, 2, 384)
  h, a = tf.split(h, [n_dims * n_caps, n_caps], -1)                   # (32, 2, 2, 368), (32, 2, 2, 16)
  h = tf.reshape(h, [batch_size, -1, n_dims, n_caps])                 # (32, 4, 23, 16)
  a = tf.nn.softmax(a, 1)
  a = tf.reshape(a, [batch_size, -1, 1, n_caps])                      # (32, 4, 1, 16)
  h = tf.reduce_sum(h * a, 1)                                         # (32, 23, 16)

  h = tf.reshape(h, [batch_size, n_caps, n_dims])                     # (32, 16, 23)

  pose, feature, presence_logit = tf.split(h, splits, -1)             # (32, 16, 6), (32, 16, 16), (32, 16, 1)
  if n_features == 0:
    feature = None

  presence_logit = tf.squeeze(presence_logit, -1)                     # (32, 16)
  if noise_scale > 0.:
    presence_logit += (
        (tf.random.uniform(presence_logit.shape) - .5) * noise_scale)

  presence = tf.nn.sigmoid(presence_logit)
  pose = geo_transform(pose)                                          # (32, 16, 6)

  return keras.Model(
      inputs=img,
      outputs=[pose, feature, presence, presence_logit]
  )


def ImageDecoder(params, pose, template_feature=None):
  '''Build templates from part capsules
  '''
  n_dims = params['n_channels']
  template_size = params['template_size']
  use_alpha_channel = params['use_alpha_channel']
  colorize_templates = params['colorize_templates']

  def make_templates(templates=None, n_templates=None, template_feature=None):
    # generate templates if it is not yet defined
    if templates is not None:
      if n_templates is not None and templates.shape[1] != n_templates:
        raise ValueError
    else:
      template_shape = ([1, n_templates] + list(template_size) + [n_dims])    # (1, 16, 11, 11, 1)
      n_els = np.prod(template_shape[2:])

      # make each templates orthogonal to each other at init
      n = max(n_templates, n_els)
      q = np.random.uniform(size=[n, n])
      q = np.linalg.qr(q)[0]
      q = q[:n_templates, :n_els].reshape(template_shape).astype(np.float32)

      q = (q - q.min()) / (q.max() - q.min())

      template_logits = tf.Variable(q)
      templates = tf.nn.relu6(template_logits * 6.) / 6.

      if use_alpha_channel:
        # templates_alpha = tf.Variable()
        pass

    if template_feature is not None and colorize_templates:
      mlp = snt.BatchApply(snt.nets.MLP([32, n_dims]))
      template_color = mlp(template_feature)[:, :, tf.newaxis, tf.newaxis]
      template_color += .99
      template_color = tf.nn.relu6(template_color * 6.) / 6.

      templates = tf.identity(templates) * template_color

    return templates                                                          # (1, 16, 11, 11, 1)

  # pose, presence=None, template_features=None, bg_image=None, img_embedding=None
  batch_size, n_templates = tf.shape(pose)[:2]

  templates = make_templates(n_templates, template_feature)                   # (1, 16, 11, 11, 1)

  if templates.shape[0] == 1:
    templates = tf.repeat(templates, batch_size, axis=0)

  warper = snt.AffineGridWarper()