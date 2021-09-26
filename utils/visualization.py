import tensorflow as tf
import matplotlib.pyplot as plt


def plot_image(x_batch, y_batch, class_names, n_img):
    maxc = 3
    r = int(n_img / (maxc + 1)) + 1
    c = int(min(maxc, n_img))

    fig, axes = plt.subplots(r, c, figsize=(5, 5))
    axes = axes.flatten()
    for x, y, ax in zip(x_batch, y_batch, axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(x[..., 0], cmap='gray')
        ax.set_title(class_names[tf.argmax(y)])
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
