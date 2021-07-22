import os

import cv2
import numpy as np
import pickle as pkl
import tensorflow as tf


# ==================================================================================================================== #
#                                            handling image                                                            #
# ==================================================================================================================== #
def load_images(img_path_list, width, height):
    num_images = len(img_path_list)
    images = np.zeros([num_images, height, width, 3], dtype=np.float32)
    for idx, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        images[idx] = cv2.resize(img/255.0, (width, height))
    return images


def load_one_image(img_path, width, height):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img / 255.0, (width, height))
    return np.array(img).astype(dtype=np.float32)


@tf.function
def random_resize_image(image, scale_range=[1.05, 1.15]):
    scale_factor = (scale_range[1] - scale_range[0]) * np.random.rand(1) + scale_range[0]
    _, height, width, _ = image.shape
    resized = [int(height * scale_factor), int(width * scale_factor)]
    return tf.image.resize(image, resized)


@tf.function
def random_crop_image(image, size):
    return tf.image.random_crop(image, size=size)


@tf.function
def random_flip_image(image):
    return tf.image.random_flip_left_right(image)


@tf.function
def random_brightness(image):
    return tf.image.random_brightness(image, max_delta=32/255)


@tf.function
def random_saturation(image):
    return tf.image.random_saturation(image, lower=0.5, upper=1.5)


@tf.function
def random_hue(image):
    return tf.image.random_hue(image, max_delta=0.2)


@tf.function
def random_contrast(image):
    return tf.image.random_contrast(image, lower=0.5, upper=1.5)


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    label = tf.cast(label, tf.int32)
    return image, label


def normalize_img_v2(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return image, label


def augmentation_v1(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)  # padding 4 to each borders
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)

    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    label = tf.cast(label, tf.int32)
    return image, label


def augmentation_v2(image, label):
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)  # padding 4 to each borders
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)

    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return image, label


def augmentation_v0_for_age(image, config):
    batch_size = image.shape[0]
    image = tf.image.resize_with_crop_or_pad(image, config.width+4, config.height+4)  # rescale within [1.05, 1.25]
    image = tf.image.random_crop(image, [batch_size, config.width, config.height, 3])
    image = tf.image.random_flip_left_right(image)  # random flip
    return image


# ==================================================================================================================== #
#                                            other utility funcs                                                       #
# ==================================================================================================================== #
def save_or_load_feature(feature_file, data_list, fdim, model, config):
    if os.path.exists(feature_file):
        features = pkl.load(open(feature_file, "rb"))
        print(f'features are loaded from {feature_file}')
    else:
        num_data = len(data_list)
        features = np.zeros((num_data, fdim), dtype=np.float32)
        num_batches = int(np.ceil(num_data / config.batch_size))
        for batch_idx in range(num_batches):
            print(f'encoding data... {batch_idx} / {num_batches}')
            start_idx = config.batch_size * batch_idx
            end_idx = min(start_idx + config.batch_size, num_data)

            batch_image = load_images(data_list[start_idx:end_idx], config.width, config.height)
            features[start_idx:end_idx, ...] = tf.squeeze(model(batch_image, training=False), axis=[1,2])

        # save feature
        pkl.dump(features, open(feature_file, "wb"))
        print(f'encoded features are saved to {feature_file}')
    return features


def save_or_load_feature_v2(feature_file, data_list, fdim, model, config):
    num_data = len(data_list)
    if os.path.exists(feature_file):
        features = pkl.load(open(feature_file, "rb"))
        print(f'features are loaded from {feature_file}')
    else:
        features = np.zeros((num_data, fdim), dtype=np.float32)
        num_batches = int(np.ceil(num_data / config.batch_size))
        for batch_idx in range(num_batches):
            print(f'encoding data... {batch_idx} / {num_batches}')
            start_idx = config.batch_size * batch_idx
            end_idx = min(start_idx + config.batch_size, num_data)

            batch_image = load_images(data_list[start_idx:end_idx], config.width, config.height)
            features[start_idx:end_idx, ...] = model(batch_image, training=False)

        # save feature
        pkl.dump(features, open(feature_file, "wb"))
        print(f'encoded features are saved to {feature_file}')
    return features


def save_or_load_for_art():
    raise NotImplementedError


def write_log(log_file, out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)
