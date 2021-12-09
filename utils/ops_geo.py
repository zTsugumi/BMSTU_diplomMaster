import tensorflow as tf


def pose_to_affine(pose):
  '''Converts pose tensor into an affine transform matrix
  Args:
    pose: rotate, scale, shift
    rotate: [..., n_caps, 2] cos_theta, sin_theta
    scale: [..., n_caps, 2] sx, sy
    shift: [..., n_caps, 2] tx, ty
  Returns:
    |sx.cos_theta -sy.sin_theta tx|
    |sx.sin_theta  sy.cos_theta ty|
  '''
  rotate, shift, scale = pose
  n_caps = rotate.shape[1]

  rotate = tf.reshape(rotate, (-1, 2))
  shift = tf.reshape(shift, (-1, 2))
  scale = tf.reshape(scale, (-1, 2))

  cos_theta = rotate[:, :1]
  sin_theta = rotate[:, 1:]
  R = tf.concat(
      (cos_theta, -sin_theta, sin_theta, cos_theta), axis=-1)
  R = tf.reshape(R, (-1, 2, 2))                   # None*n_caps, 2, 2

  S = tf.linalg.diag(scale)                       # None*n_caps, 2, 2

  T = tf.reshape(shift, (-1, 2, 1))               # None*n_caps, 2, 1

  R_S = tf.matmul(R, S)                           # None*n_caps, 2, 2
  affine = tf.concat((R_S, T), axis=-1)           # None*n_caps, 2, 3

  # last_row = tf.constant([[[0, 0, 1]]], dtype=affine.dtype)
  # last_row = tf.repeat(last_row, tf.shape(affine)[0], axis=0)
  # affine = tf.concat((affine, last_row), axis=1)  # None*n_caps, 3, 3

  affine = tf.reshape(affine, (-1, n_caps, 2, 3))

  # None, n_caps, 2, 3
  return affine


def affine_inverse(affine):
  '''Inverts affine matrix
  Args: 
    affine: [..., n_caps, 2, 3]
    ->  |sx.cos_theta -sy.sin_theta tx|
        |sx.sin_theta  sy.cos_theta ty|
  Returns:
    affine ^ -1
  '''
  R = affine[:, :, :, :2]
  T = affine[:, :, :, 2:]

  R_inv = tf.linalg.inv(R)
  T_inv = -tf.matmul(R_inv, T)

  # None, n_caps, 2, 3
  return tf.concat([R_inv, T_inv], axis=-1)


def transform_affine(capsules, affine, out_size):
  '''
  Spatial Transformer on capsule
  Args:
    capsules: [...*n_caps, 128, 128, 3]
    theta: [...*n_caps, 2, 3]
    ->  |sx.cos_theta -sy.sin_theta tx|
        |sx.sin_theta  sy.cos_theta ty|
  Returns:
    warped_capsule: [...*n_caps, 128, 128, 3]
  '''

  def _grid_generator(b, h, w):
    x = tf.linspace(-1, 1, w)
    y = tf.linspace(-1, 1, h)

    xx, yy = tf.meshgrid(x, y)
    xx = tf.reshape(xx, (-1,))
    yy = tf.reshape(yy, (-1,))
    homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)])
    homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0)
    homogenous_coordinates = tf.tile(homogenous_coordinates, [b, 1, 1])
    homogenous_coordinates = tf.cast(homogenous_coordinates, dtype=tf.float32)
    return homogenous_coordinates

  def _advanced_indexing(capsules, out_size, x, y):
    shape = tf.shape(capsules)
    batch_size = shape[0]
    out_h, out_w = out_size

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, out_h, out_w))
    indices = tf.stack([b, y, x], 3)
    return tf.gather_nd(capsules, indices)

  def _interpolate(capsules, x_s, y_s, out_size):
    out_h, out_w = out_size

    # Variable Casting
    x0 = tf.cast(tf.math.floor(x_s), dtype=tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.math.floor(y_s), dtype=tf.int32)
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, out_w - 1)
    x1 = tf.clip_by_value(x1, 0, out_w - 1)
    y0 = tf.clip_by_value(y0, 0, out_h - 1)
    y1 = tf.clip_by_value(y1, 0, out_h - 1)
    x_s = tf.clip_by_value(x_s, 0, tf.cast(out_w, dtype=tf.float32) - 1.0)
    y_s = tf.clip_by_value(y_s, 0, tf.cast(out_h, dtype=tf.float32) - 1.0)

    # Advanced indexing
    Ia = _advanced_indexing(capsules, out_size, x0, y0)
    Ib = _advanced_indexing(capsules, out_size, x0, y1)
    Ic = _advanced_indexing(capsules, out_size, x1, y0)
    Id = _advanced_indexing(capsules, out_size, x1, y1)

    # Interpolation
    x0 = tf.cast(x0, dtype=tf.float32)
    x1 = tf.cast(x1, dtype=tf.float32)
    y0 = tf.cast(y0, dtype=tf.float32)
    y1 = tf.cast(y1, dtype=tf.float32)

    wa = (x1 - x_s) * (y1 - y_s)
    wb = (x1 - x_s) * (y_s - y0)
    wc = (x_s - x0) * (y1 - y_s)
    wd = (x_s - x0) * (y_s - y0)

    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    return tf.math.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

  def _transform(capsules, affine, out_size):
    shape = tf.shape(capsules)
    b = shape[0]
    c = capsules.shape[-1]
    out_h, out_w = out_size

    # capsules = tf.reshape(capsules, (-1, h, w, c))  # None*n_caps, 128, 128, 3
    # affine = tf.reshape(affine, (-1, 2, 3))         # None*n_caps, 2, 3

    # Generate grid (x_t, y_t, 1)
    grid = _grid_generator(b, out_h, out_w)         # None, 128, 128, 1

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    transformed = tf.matmul(affine, grid)
    transformed = tf.transpose(transformed, perm=[0, 2, 1])
    transformed = tf.reshape(transformed, [-1, out_h, out_w, 2])

    x_t = transformed[:, :, :, 0]
    y_t = transformed[:, :, :, 1]

    x_s = ((x_t + 1.) * tf.cast(out_w, dtype=tf.float32)) * 0.5
    y_s = ((y_t + 1.) * tf.cast(out_h, dtype=tf.float32)) * 0.5

    # Interpolate
    transformed = _interpolate(capsules, x_s, y_s, out_size)
    transformed = tf.reshape(transformed, (-1, out_h, out_w, c))

    return transformed

  output = _transform(capsules, affine, out_size)
  return output


def transform_occlusion(capsules, occlusion):
  '''
  Apply occlusion as softmax
  Args:
    capsules: [..., n_caps, 128, 128, 3]
    occlusion: [..., n_caps]    
  Returns:
    warped_capsule: [..., n_caps, 128, 128, 3]
  '''
  shape = tf.shape(capsules)
  b = shape[0]
  n_caps, h, w, c = capsules.shape[1:]

  capsules = tf.reshape(capsules, (b, n_caps, -1))              # None, n_caps, 128*128*3
  occlusion = tf.reshape(occlusion, (b, n_caps, 1))             # None, n_caps, 1

  # capsules, bg = tf.split(capsules, [n_caps - 1, 1], axis=1)
  # occlusion, bg_occlusion = tf.split(occlusion, [n_caps - 1, 1], axis=1)

  # WIP: softmax before mul or vice versa??
  capsules_occluded = capsules * occlusion
  occlusion = tf.nn.softmax(capsules_occluded, axis=1)
  # bg_occluded = bg * bg_occlusion

  capsules_merged = tf.reduce_sum(capsules_occluded, axis=1)    # None, 128*128*3
  capsules_merged = tf.reshape(capsules_merged, (-1, h, w, c))  # None, 128, 128, 3

  return capsules_merged
