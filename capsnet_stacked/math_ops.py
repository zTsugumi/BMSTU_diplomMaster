import math
import tensorflow as tf


def geo_transform(pose, nonlinear=True, as_matrix=False):
  '''Convers pose tensor into an affine or similarity transform
  Args:
    pose_tensor: [..., 6] tensor
    similarity: bool
    nonlinear: bool; applies nonlinearities to pose params if True
    as_matrix: bool; convers the transform to a matrix if True

  Returns:
    [..., 3, 3] tensor if `as_matrix` else [..., 6] tensor
  '''
  scale_x, scale_y, theta, shear, trans_x, trans_y = tf.split(pose, 6, -1)

  if nonlinear:
    scale_x, scale_y = (tf.nn.sigmoid(i) + 1e-2
                        for i in (scale_x, scale_y))

    trans_x, trans_y, shear = (
        tf.nn.tanh(i * 5.) for i in (trans_x, trans_y, shear))

    theta *= 2. * math.pi

  else:
    scale_x, scale_y = (abs(i) + 1e-2 for i in (scale_x, scale_y))

  c, s = tf.cos(theta), tf.sin(theta)

  pose = [
      scale_x * c + shear * scale_y * s, -scale_x * s + shear * scale_y * c,
      trans_x, scale_y * s, scale_y * c, trans_y
  ]

  pose = tf.concat(pose, -1)

  # convert to a matrix
  if as_matrix:
    shape = pose.shape[:-1].as_list()
    shape += [2, 3]
    pose = tf.reshape(pose, shape)
    zeros = tf.zeros_like(pose[..., :1, 0])
    last = tf.stack([zeros, zeros, zeros + 1], -1)
    pose = tf.concat([pose, last], -2)

  return pose
