import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt


def plot_image(x_batch, y_batch, class_names, n_img, mode='supervised'):
  maxc = 3
  r = int(n_img / (maxc + 1)) + 1
  c = int(min(maxc, n_img))

  _, axes = plt.subplots(r, c, figsize=(5, 5))
  axes = axes.flatten()

  if mode == 'supervised':
    for x, y, ax in zip(x_batch, y_batch, axes):
      ax.set_xticks([])
      ax.set_yticks([])
      ax.imshow(x[..., 0], cmap='gray')
      ax.set_title(class_names[tf.argmax(y)])
    plt.tight_layout()
    plt.show()
  elif mode == 'unsupervised':
    for x, ax in zip(x_batch, axes):
      ax.set_xticks([])
      ax.set_yticks([])
      ax.imshow(x)
    plt.tight_layout()
    plt.show()


def plot_image_misclass(x, y_true, y_pred, class_names, n_img=12):
  idc = tf.squeeze(
      tf.where(tf.argmax(y_pred, axis=-1) != tf.argmax(y_true, axis=-1)),
      axis=-1)

  if tf.equal(tf.size(idc), 0):
    return

  n_img = min(n_img, tf.size(idc))
  maxc = 4
  r = int(n_img / (maxc + 1)) + 1
  c = int(min(maxc, n_img))

  fig, axes = plt.subplots(r, c, figsize=(10, 5), squeeze=False)
  axes = axes.flatten()

  if x.shape[-1] == 2:    # for SmallNORB
    xs = x[..., 0]
  else:
    xs = x

  for idx, ax in zip(idc, axes):
    ax.imshow(xs[idx, ...], cmap='gray')
    ax.set_axis_off()
    idx_true = tf.argmax(y_true[idx])
    class_true = class_names[idx_true]
    class_true_prob = y_pred[idx][idx_true]
    idx_pred = tf.argmax(y_pred[idx])
    class_pred = class_names[idx_pred]
    class_pred_prob = y_pred[idx][idx_pred]
    ax.set_title(
        f'True class: {class_true} - {class_true_prob*100:.4f}%\nPredicted class: {class_pred} - {class_pred_prob*100:.4f}%')
  plt.tight_layout()
  plt.show()


def plot_filters_stimuli(model, layer, is_conv=True, iters=20, lr=1.):
  def find_max_stimuli(model, layer, is_conv, filter_index, iters=20, lr=1.0, verbose=True):
    def init_img():
      '''
      Random an image
      '''
      img_w, img_h, img_c = model.input.shape.as_list()[1:4]
      img = tf.random.uniform((1, img_w, img_h, img_c))
      return (img - 0.5) * (0.1 if is_conv else 0.001)

    def compute_loss(input_image, filter_index):
      '''
      Loss: Mean of the activation of a specific filter in out target layer
      '''
      # feature_extractor: return the activation values for our target layer
      feature_extractor = keras.Model(
          inputs=layer.input, outputs=layer.output)

      activation = feature_extractor(input_image)

      if is_conv:
        # Avoid border artifacts by only involving non-border pixels in the loss.
        filter_activation = activation[:, 2:-2, 2:-2, filter_index]
      else:
        filter_activation = activation[:, filter_index]

      return tf.reduce_mean(filter_activation)

    # @tf.function
    def gradient_ascent_step(img, filter_index, lr):
      '''
      Calculate one single step of gradient ascent
      '''
      with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)

      grads = tape.gradient(loss, img)
      grads = tf.math.l2_normalize(grads)
      img += lr * grads

      return loss, img

    def deprocess_img(img):
      # Normalize array: center on 0., ensure variance is 0.15
      img -= img.mean()
      img /= img.std() + 1e-5
      # img = (img - np.mean(x_train2)) / np.std(x_train2)

      # Center crop
      # img = img[25:-25, 25:-25, :]

      # Clip to [0, 1]
      img += 0.5
      img = np.clip(img, 0, 1)

      # Convert to RGB
      img *= 255
      img = np.clip(img, 0, 255).astype('uint8')

      return img

    img = init_img()
    for iter in range(iters):
      loss, img = gradient_ascent_step(img, filter_index, lr)
      if verbose:
        print(f'Loss value on {iter}: {loss}')
    img = deprocess_img(img[0].numpy())

    return loss, img

  num_filter = layer.output.shape.as_list()[-1]

  maxc = 3
  rows = int(num_filter / (maxc + 1)) + 1
  cols = int(min(maxc, num_filter))

  ncol = 8
  nrow = num_filter // ncol + 1
  fig = plt.figure(figsize=(3 * cols - 1, 6 * rows - 1))

  for filter in range(num_filter):
    lost, img = find_max_stimuli(
        model, layer, is_conv=is_conv,
        filter_index=filter, iters=iters, lr=lr, verbose=False)
    ax = fig.add_subplot(nrow, ncol, filter + 1)
    ax.grid('off')
    ax.axis('off')
    ax.set_title(f"Filter #{filter}")
    ax.imshow(img)

  plt.show()
