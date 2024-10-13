import tensorflow as tf
import numpy as np
import cv2


# %%
img = np.random.random_integers(0, 255, (5, 5))
mask = np.random.random_integers(0, 1, (5, 5)) * 255

cv2.imwrite("img_test.png", img)
cv2.imwrite("mask_test.png", mask)

image_path = "./img_test.png"
mask_path = "./mask_test.png"

image_list_train = [image_path]
mask_list_train = [mask_path]

dataset_train = tf.data.Dataset.from_tensor_slices((image_list_train, mask_list_train))

# %%
IMG_SIZE = (96, 128)

def test_fun(mask):
    return(mask)

def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels = 3)
    mask = tf.math.reduce_max(mask, axis = -1, keepdims = True)
    return img, mask


def preprocess(image, mask):
    input_image = tf.image.resize(image, (IMG_SIZE[0], IMG_SIZE[1]), method = 'nearest')
    input_mask = tf.image.resize(mask, (IMG_SIZE[0], IMG_SIZE[1]), method = 'nearest')

    input_image = input_image / 255.

    input_mask = tf.py_function(test_fun, [input_mask], np.uint8)
    input_mask.set_shape(tf.TensorShape([IMG_SIZE[0], IMG_SIZE[1], 1]))
    print(input_mask.shape)
    print(type(input_mask))
    return input_image, input_mask


# %%
image_train_ds = dataset_train.map(process_path)
processed_image_train_ds = image_train_ds.map(preprocess)
processed_image_train_ds


# %%

mask_path = list(processed_image_train_ds.take(1))