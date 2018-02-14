import os
import numpy as np


def load_image_path_pairs(src_img_dir_path, dest_img_dir_path):

    src_images = dict()
    dest_images = dict()
    for f in os.listdir(src_img_dir_path):
        filepath = os.path.join(src_img_dir_path, f)
        if os.path.isfile(filepath) and f.endswith('.png'):
            name = f.replace('.png', '')
            src_images[name] = filepath
    for f in os.listdir(dest_img_dir_path):
        filepath = os.path.join(dest_img_dir_path, f)
        if os.path.isfile(filepath) and f.endswith('.png'):
            name = f.replace('.png', '')
            dest_images[name] = filepath

    result = []
    for name, img_path in src_images.items():
        if name in dest_images:
            dest_file_path = dest_images[name]
            result.append([img_path, dest_file_path])

    return np.array(result)
