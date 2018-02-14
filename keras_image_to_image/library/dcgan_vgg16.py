from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Input, Dense, Reshape, concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, img_to_array

from keras_image_to_image.library.utility.image_loader import load_and_scale_image
from keras_image_to_image.library.utility.image_utils import combine_normalized_images, img_from_normalized_img
from keras import backend as K
import numpy as np
from PIL import Image
import os


class DCGanWithVGG16(object):
    model_name = 'dc-gan'

    def __init__(self):
        K.set_image_dim_ordering('tf')
        self.generator = None
        self.discriminator = None
        self.model = None
        self.img_width = 7
        self.img_height = 7
        self.img_channels = 1
        self.random_input_dim = 100
        self.img_input_dim = 1000
        self.config = None
        self.vgg16_model = None

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, DCGanWithVGG16.model_name + '-config.npy')

    @staticmethod
    def get_weight_file_path(model_dir_path, model_type):
        return os.path.join(model_dir_path, DCGanWithVGG16.model_name + '-' + model_type + '-weights.h5')

    def create_model(self):
        init_img_width = self.img_width // 4
        init_img_height = self.img_height // 4

        random_input = Input(shape=(self.random_input_dim,))
        img_input1 = Input(shape=(self.img_input_dim,))
        random_dense = Dense(1024)(random_input)
        text_layer1 = Dense(1024)(img_input1)

        merged = concatenate([random_dense, text_layer1])
        generator_layer = Activation('tanh')(merged)

        generator_layer = Dense(128 * init_img_width * init_img_height)(generator_layer)
        generator_layer = BatchNormalization()(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = Reshape((init_img_width, init_img_height, 128),
                                  input_shape=(128 * init_img_width * init_img_height,))(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(64, kernel_size=5, padding='same')(generator_layer)
        generator_layer = Activation('tanh')(generator_layer)
        generator_layer = UpSampling2D(size=(2, 2))(generator_layer)
        generator_layer = Conv2D(self.img_channels, kernel_size=5, padding='same')(generator_layer)
        generator_output = Activation('tanh')(generator_layer)

        self.generator = Model([random_input, img_input1], generator_output)

        self.generator.compile(loss='mean_squared_error', optimizer="SGD")

        print('generator: ', self.generator.summary())

        text_input2 = Input(shape=(self.img_input_dim,))
        text_layer2 = Dense(1024)(text_input2)

        img_input2 = Input(shape=(self.img_width, self.img_height, self.img_channels))
        img_layer2 = Conv2D(64, kernel_size=(5, 5), padding='same')(
            img_input2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Conv2D(128, kernel_size=5)(img_layer2)
        img_layer2 = Activation('tanh')(img_layer2)
        img_layer2 = MaxPooling2D(pool_size=(2, 2))(img_layer2)
        img_layer2 = Flatten()(img_layer2)
        img_layer2 = Dense(1024)(img_layer2)

        merged = concatenate([img_layer2, text_layer2])

        discriminator_layer = Activation('tanh')(merged)
        discriminator_layer = Dense(1)(discriminator_layer)
        discriminator_output = Activation('sigmoid')(discriminator_layer)

        self.discriminator = Model([img_input2, text_input2], discriminator_output)

        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

        print('discriminator: ', self.discriminator.summary())

        model_output = self.discriminator([self.generator.output, img_input1])

        self.model = Model([random_input, img_input1], model_output)
        self.discriminator.trainable = False

        g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=g_optim)

        print('generator-discriminator: ', self.model.summary())

        if self.vgg16_model is None:
            self.vgg16_model = VGG16(include_top=True, weights='imagenet')
            self.vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

    def load_model(self, model_dir_path):
        config_file_path = DCGanWithVGG16.get_config_file_path(model_dir_path)
        self.config = np.load(config_file_path).item()
        self.img_width = self.config['img_width']
        self.img_height = self.config['img_height']
        self.img_channels = self.config['img_channels']
        self.random_input_dim = self.config['random_input_dim']
        self.create_model()
        self.generator.load_weights(DCGanWithVGG16.get_weight_file_path(model_dir_path, 'generator'))
        self.discriminator.load_weights(DCGanWithVGG16.get_weight_file_path(model_dir_path, 'discriminator'))

    def fit(self, model_dir_path, image_path_pairs, epochs=None, batch_size=None, snapshot_dir_path=None,
            snapshot_interval=None):
        if epochs is None:
            epochs = 100

        if batch_size is None:
            batch_size = 128

        if snapshot_interval is None:
            snapshot_interval = 20

        self.config = dict()
        self.config['img_width'] = self.img_width
        self.config['img_height'] = self.img_height
        self.config['random_input_dim'] = self.random_input_dim
        self.config['img_channels'] = self.img_channels

        config_file_path = DCGanWithVGG16.get_config_file_path(model_dir_path)

        np.save(config_file_path, self.config)
        noise = np.zeros((batch_size, self.random_input_dim))
        src_image_feature_batch = np.zeros((batch_size, self.img_input_dim))

        self.create_model()

        if self.vgg16_model is None:
            self.vgg16_model = VGG16(include_top=True, weights='imagenet')
            self.vgg16_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

        for epoch in range(epochs):
            print("Epoch is", epoch)
            batch_count = int(image_path_pairs.shape[0] / batch_size)
            print("Number of batches", batch_count)
            for batch_index in range(batch_count):
                # Step 1: train the discriminator

                image_path_pair_batch = image_path_pairs[batch_index * batch_size:(batch_index + 1) * batch_size]

                dest_image_batch = []
                for index in range(batch_size):
                    image_path_pair = image_path_pair_batch[index]
                    src_image_path = image_path_pair[0]
                    dest_image_path = image_path_pair[1]
                    dest_image_batch.append(load_and_scale_image(dest_image_path, self.img_width, self.img_height))
                    src_image_feature_batch[index, :] = self.encode_src_image(src_image_path)
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)

                dest_image_batch = np.array(dest_image_batch)

                # image_batch = np.transpose(image_batch, (0, 2, 3, 1))
                generated_images = self.generator.predict([noise, src_image_feature_batch], verbose=0)

                if (epoch * batch_size + batch_index) % snapshot_interval == 0 and snapshot_dir_path is not None:
                    self.save_snapshots(generated_images, snapshot_dir_path=snapshot_dir_path,
                                        epoch=epoch, batch_index=batch_index)

                self.discriminator.trainable = True
                d_loss = self.discriminator.train_on_batch([np.concatenate((dest_image_batch, generated_images)),
                                                            np.concatenate((src_image_feature_batch, src_image_feature_batch))],
                                                           np.array([1] * batch_size + [0] * batch_size))
                print("Epoch %d batch %d d_loss : %f" % (epoch, batch_index, d_loss))

                # Step 2: train the generator
                for index in range(batch_size):
                    noise[index, :] = np.random.uniform(-1, 1, self.random_input_dim)
                self.discriminator.trainable = False
                g_loss = self.model.train_on_batch([noise, src_image_feature_batch], np.array([1] * batch_size))

                print("Epoch %d batch %d g_loss : %f" % (epoch, batch_index, g_loss))
                if (epoch * batch_size + batch_index) % 10 == 9:
                    self.generator.save_weights(DCGanWithVGG16.get_weight_file_path(model_dir_path, 'generator'), True)
                    self.discriminator.save_weights(DCGanWithVGG16.get_weight_file_path(model_dir_path, 'discriminator'), True)

        self.generator.save_weights(DCGanWithVGG16.get_weight_file_path(model_dir_path, 'generator'), True)
        self.discriminator.save_weights(DCGanWithVGG16.get_weight_file_path(model_dir_path, 'discriminator'), True)

    def encode_src_image(self, src_img_path):
        img_input = img_to_array(load_img(src_img_path, target_size=(224, 224)))
        img_input = np.expand_dims(img_input, axis=0)
        img_input = preprocess_input(img_input)
        return self.vgg16_model.predict(img_input).ravel()

    def generate_image_from_image_file(self, src_image_path):
        noise = np.zeros(shape=(1, self.random_input_dim))
        encoded_text = np.zeros(shape=(1, self.img_input_dim))
        encoded_text[0, :] = self.encode_src_image(src_image_path)
        noise[0, :] = np.random.uniform(-1, 1, self.random_input_dim)
        generated_images = self.generator.predict([noise, encoded_text], verbose=0)
        generated_image = generated_images[0]
        generated_image = generated_image * 127.5 + 127.5
        return Image.fromarray(generated_image.astype(np.uint8))

    def save_snapshots(self, generated_images, snapshot_dir_path, epoch, batch_index):
        image = combine_normalized_images(generated_images)
        img_from_normalized_img(image).save(
            os.path.join(snapshot_dir_path, DCGanWithVGG16.model_name + '-' + str(epoch) + "-" + str(batch_index) + ".png"))
