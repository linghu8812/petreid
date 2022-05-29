import os.path as osp

from PIL import Image
import numpy as np


def read_image(path):
    """Reads image from path using ``PIL.Image``.
    Args:
        path (str): path to an image.
    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert("RGB")
            got_img = True
        except IOError:
            print(
                f'IOError incurred when reading "{path}". '
                f"Will redo. Don't worry. Just chill."
            )
    src_img = np.zeros((max(img.size), max(img.size), 3)).astype(np.uint8)
    src_img[:img.size[1], :img.size[0], :] = np.array(img)
    img = Image.fromarray(src_img)
    return img


def save_image(image_numpy, path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(path)
