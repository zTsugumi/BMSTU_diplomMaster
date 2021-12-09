import os
import numpy as np
from skimage import io, img_as_float32, color
from imageio import mimread


def load_video(
        dir='./data/voxceleb',
        frame_dim=(128, 128)):
  if os.path.isdir(dir):
    frames = sorted(os.listdir(dir))
    n_frames = len(frames)
    vid_array = np.array(
        [img_as_float32(io.imread(os.path.join(dir, frames[i]))) for i in range(n_frames)]
    )
  elif dir.lower().endswith('.png') or dir.lower().endswith('.jpg'):
    frame = io.imread(dir)

    if len(frame.shape) == 2 or frame.shape[2] == 1:
      frame = color.gray2rgb(frame)

    if frame.shape[2] == 4:
      frame = frame[..., :3]

    frame = img_as_float32(frame)

    vid_array = np.moveaxis(frame, 1, 0)
    vid_array = vid_array.reshape((-1,)+frame_dim)
    vid_array = np.moveaxis(vid_array, 1, 2)
  elif dir.lower().endswith('.gif') or dir.lower().endswith('.mp4') or dir.lower().endswith('.mov'):
    vid = np.array(mimread(dir))

    if len(vid.shape) == 3:
      vid_array = np.array([color.gray2rgb(frame) for frame in vid])

    if vid.shape[-1] == 4:
      vid_array = vid[..., :3]

    vid_array = img_as_float32(vid_array)
  else:
    raise Exception(f'Unknown file extensions {dir}')

  return vid_array

