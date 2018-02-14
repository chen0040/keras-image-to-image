from keras_image_to_image.library.dcgan_vgg16 import DCGanWithVGG16
from keras_image_to_image.library.utility.img_directory_loader import load_image_path_pairs
import numpy as np
from random import shuffle


def main():
    seed = 42

    np.random.seed(seed)

    src_img_dir_path = './data/pokemon/src_img'
    dest_dir_path = './data/pokemon/dest_img'
    model_dir_path = './models'

    img_width = 32
    img_height = 32
    img_channels = 3

    image_path_pairs = load_image_path_pairs(src_img_dir_path, dest_dir_path)

    shuffle(image_path_pairs)

    gan = DCGanWithVGG16()
    gan.img_width = img_width
    gan.img_height = img_height
    gan.img_channels = img_channels
    gan.random_input_dim = 200

    batch_size = 16
    epochs = 1000
    gan.fit(model_dir_path=model_dir_path, image_path_pairs=image_path_pairs,
            snapshot_dir_path='./data/snapshots',
            snapshot_interval=100,
            batch_size=batch_size,
            epochs=epochs)


if __name__ == '__main__':
    main()
