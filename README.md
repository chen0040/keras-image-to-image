# keras-image-to-image

Transform one image to another image in Keras using GAN

* [dcgan_vgg16.py](keras_image_to_image/library/dcgan_vgg16.py): this version has a very noisy input with text input (half of the 
input is pure noise while the other half is generated from VGG16 encoding of source image)
* [dcgan_vgg16_v2.py](keras_image_to_image/library/dcgan_vgg16_v2.py): this version remove noise as input (the input is just 
VGG16 encoding of source image)
* [dcgan_vgg16_v3.py](keras_image_to_image/library/dcgan_vgg16_v3.py): this version add a configurable amount of noise as input 
together with the VGG16 encoding of source image

# Usage

The sample codes below only generate very small images, but the image size can be increased if you have sufficient
memory 

### Image-to-Image using VGG16 and Deep Convolution GAN

Below is the [sample codes](demo/dcgan_vgg16_train.py) to train the DCGan on a set of pokemon samples of image pair 
(src_image_path, dest_image_path)

```python
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
    gan.random_input_dim = 100

    batch_size = 16
    epochs = 2000
    gan.fit(model_dir_path=model_dir_path, image_path_pairs=image_path_pairs,
            snapshot_dir_path='./data/snapshots',
            snapshot_interval=500,
            batch_size=batch_size,
            epochs=epochs)


if __name__ == '__main__':
    main()

```

Below is the [sample codes](demo/dcgan_vgg16_generate.py) on how to load the trained DCGan model to generate
3 new pokemon image samples from a pokemon source image:

```python
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

        image = Image.open(dest_image_path).resize((img_width, img_height), PIL.Image.ANTIALIAS)
        image.save('./data/outputs/' + DCGanWithVGG16.model_name + '-generated-' + str(i) + '-0.png')
        for j in range(3):
            generated_image = gan.generate_image_from_image_file(src_image_path)
            generated_image.save('./data/outputs/' + DCGanWithVGG16.model_name + '-generated-' + str(i) + '-' + str(j) + '.png')


if __name__ == '__main__':
    main()

```

# Configure to run on GPU on Windows

* Step 1: Change tensorflow to tensorflow-gpu in requirements.txt and install tensorflow-gpu
* Step 2: Download and install the [CUDA® Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Please note that
currently CUDA® Toolkit 9.1 is not yet supported by tensorflow, therefore you should download CUDA® Toolkit 9.0)
* Step 3: Download and unzip the [cuDNN 7.4 for CUDA@ Toolkit 9.0](https://developer.nvidia.com/cudnn) and add the
bin folder of the unzipped directory to the $PATH of your Windows environment 

