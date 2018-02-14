import PIL

from keras_image_to_image.library.dcgan_vgg16 import DCGanWithVGG16
from keras_image_to_image.library.utility.img_directory_loader import load_image_path_pairs
import numpy as np
from random import shuffle
from PIL import Image

def main():
    seed = 42

    np.random.seed(seed)

    img_dir_path = './data/pokemon/src_img'
    txt_dir_path = './data/pokemon/dest_img'
    model_dir_path = './models'

    img_width = 32
    img_height = 32

    image_label_pairs = load_image_path_pairs(img_dir_path, txt_dir_path)

    shuffle(image_label_pairs)

    gan = DCGanWithVGG16()
    gan.load_model(model_dir_path)

    for i in range(3):
        image_path_pair = image_label_pairs[i]
        src_image_path = image_path_pair[0]
        dest_image_path = image_path_pair[1]

        image = Image.open(dest_image_path).img.resize((img_width, img_height), PIL.Image.ANTIALIAS)
        image.save('./data/outputs/' + DCGanWithVGG16.model_name + '-generated-' + str(i) + '-0.png')
        for j in range(3):
            generated_image = gan.generate_image_from_image_file(src_image_path)
            generated_image.save('./data/outputs/' + DCGanWithVGG16.model_name + '-generated-' + str(i) + '-' + str(j) + '.png')


if __name__ == '__main__':
    main()