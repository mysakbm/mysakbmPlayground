import tensorflow as tf
import numpy as np
import os
import pandas as pd
import imageio as iio
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import glob
import cv2
import random
import src.utility as util

# pd.get_option("display.max_columns")
# pd.get_option("display.max_rows")

pd.set_option("display.max_rows", 60)
pd.set_option("display.max_columns", 60)
pd.set_option("display.precision", 4)
pd.set_option("expand_frame_repr", True)


# %%
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Cropping2D

from test_utils import summary, comparator

# %%

image_path = os.path.join('../data/stamps_datasets/staver/data/augmented/imgs/')
mask_path = os.path.join('../data/stamps_datasets/staver/data/augmented/masks/')

# image_path = os.path.join('../data/carla/data/CameraRGB/')
# mask_path = os.path.join('../data/carla/data/CameraSeg/')

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

folder_to_save_weights = "./logs/" + timestamp + "/weights/"
os.makedirs(folder_to_save_weights, exist_ok = True)

folder_to_save_results = "./logs/" + timestamp + "/"
os.makedirs(folder_to_save_results, exist_ok = True)

#%%
image_list = sorted(os.listdir(image_path))
mask_list = sorted(os.listdir(mask_path))

idx = list(range(len(image_list)))
random.shuffle(idx)
train_idx = idx[:-100]
valid_idx = idx[-100:-20]
test_idx = idx[-80:]

image_list_train = [image_path + image_list[i] for i in train_idx]
mask_list_train = [mask_path + mask_list[i] for i in train_idx]

image_list_valid = [image_path + image_list[i] for i in valid_idx]
mask_list_valid = [mask_path + mask_list[i] for i in valid_idx]

image_list_test = [image_path + image_list[i] for i in test_idx]
mask_list_test = [mask_path + mask_list[i] for i in test_idx]

# %%

N = 0
img = iio.imread(image_list_train[N])
mask = iio.imread(mask_list_train[N])

fig, arr = plt.subplots(1, 2, figsize = (14, 10))
arr[0].imshow(img)
arr[0].set_title('Image')
arr[1].imshow(mask[:, :, 0])
arr[1].set_title('Segmentation')
plt.show()

#%%
dataset_train = tf.data.Dataset.from_tensor_slices((image_list_train, mask_list_train))
dataset_valid = tf.data.Dataset.from_tensor_slices((image_list_valid, mask_list_valid))
dataset_test = tf.data.Dataset.from_tensor_slices((image_list_test, mask_list_test))

# %% OLD CODE
# # %% TRAIN
#
# image_list_train_ds = tf.data.Dataset.list_files(image_list_train, shuffle = False)
# mask_list_train_ds = tf.data.Dataset.list_files(mask_list_train, shuffle = False)
#
# for path in zip(image_list_train_ds.take(3), mask_list_train_ds.take(3)):
#     print(path)
#
# image_filenames = tf.constant(image_list_train)
# masks_filenames = tf.constant(mask_list_train)
#
# dataset_train = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
#
# for image, mask in dataset_train.take(1):
#     print(image)
#     print(mask)
#
# #%% VALID
#
# image_list_valid_ds = tf.data.Dataset.list_files(image_list_valid, shuffle = False)
# mask_list_valid_ds = tf.data.Dataset.list_files(mask_list_valid, shuffle = False)
#
# for path in zip(image_list_valid_ds.take(3), mask_list_valid_ds.take(3)):
#     print(path)
#
# image_filenames = tf.constant(image_list_valid)
# masks_filenames = tf.constant(mask_list_valid)
#
# dataset_valid = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
#
# for image, mask in dataset_valid.take(1):
#     print(image)
#     print(mask)
#
# # %% Test
# image_filenames = tf.constant(image_list_test)
# masks_filenames = tf.constant(mask_list_test)
# dataset_test = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))
#
#
# def process_path(image_path, mask_path):
#     img = tf.io.read_file(image_path)
#     img = tf.image.decode_png(img, channels = 3)
#     img = tf.image.convert_image_dtype(img, tf.float32)
#
#     mask = tf.io.read_file(mask_path)
#     mask = tf.image.decode_png(mask, channels = 3)
#     mask = tf.math.reduce_max(mask, axis = -1, keepdims = True)
#     return img, mask
#
#
# def preprocess(image, mask):
#     input_image = tf.image.resize(image, (96, 128), method = 'nearest')
#     input_mask = tf.image.resize(mask, (96, 128), method = 'nearest')
#
#     input_image = input_image / 255.
#
#     return input_image, input_mask
# %%

# IMG_SIZE = (1632, 2304)
# IMG_SIZE = (816, 1152)
# IMG_SIZE = (96, 128)
IMG_SIZE = (608, 800)

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

    # For the mask, do image binarization (not need in general, this is for testing purposes
    input_mask = tf.numpy_function(util.image_binarization, [input_mask], np.float32)[:, :, None]
    input_mask.set_shape(tf.TensorShape([IMG_SIZE[0], IMG_SIZE[1], 1]))
    return input_image, input_mask


# %%
image_train_ds = dataset_train.map(process_path)
processed_image_train_ds = image_train_ds.map(preprocess)
processed_image_train_ds

image_valid_ds = dataset_valid.map(process_path)
processed_image_valid_ds = image_valid_ds.map(preprocess)

image_test_ds = dataset_test.map(process_path)
processed_image_test_ds = image_test_ds.map(preprocess)

# %%

mask_path = list(dataset_train.take(1))[0][1]
mask = tf.io.read_file(mask_path)
mask = tf.image.decode_png(mask, channels=3)
mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
mask = tf.numpy_function(util.image_binarization, [mask], np.uint8)[:, :, None]
mask = tf.image.resize(mask, (IMG_SIZE[0], IMG_SIZE[1]), method = 'nearest')
np.unique(mask, return_counts=True)


# %%

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# %%
for image, mask in image_train_ds.take(1):
    sample_image, sample_mask = image, mask
    print(mask.shape)
    display([sample_image, sample_mask])

# %%
for image, mask in processed_image_train_ds.take(1):
    sample_image, sample_mask = image, mask
    print(mask.shape)
    display([sample_image, sample_mask])

# %%
def double_conv_block(x, n_filters):
    x = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x

# pyimagesearch version https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = MaxPooling2D(pool_size = (2, 2))(f)
    p = Dropout(0.3)(p)
    return f, p

#%%

# pyimagesearch version https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
def upsample_block(x, conv_features, n_filters):
    # upsample
    x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = concatenate([x, conv_features])
    # dropout
    x = Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x

# Coursera Version
# def upsample_block(x, conv_features, n_filters):
#     # upsample
#     x = Conv2DTranspose(n_filters, 3, strides = (2, 2), padding="same")(x)
#     x = concatenate([x, conv_features])
#     x = double_conv_block(x, n_filters)
#     return x

# %%
def unet_model(input_size = (96, 128, 3), n_filters = 32, n_classes = 23):
    """
    Unet model

    Arguments:
        input_size -- Input shape
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns:
        model -- tf.keras.Model
    """
    inputs = Input(input_size)
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, n_filters) # 64
    # 2 - downsample
    f2, p2 = downsample_block(p1, n_filters * 2) # 128
    # 3 - downsample
    f3, p3 = downsample_block(p2, n_filters * 4) # 256
    # 4 - downsample
    f4, p4 = downsample_block(p3, n_filters * 8) # 512

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, n_filters * 16) # 1024

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, n_filters * 8) # 512
    # 7 - upsample
    u7 = upsample_block(u6, f3, n_filters * 4) # 256
    # 8 - upsample
    u8 = upsample_block(u7, f2, n_filters * 2) # 128
    # 9 - upsample
    u9 = upsample_block(u8, f1, n_filters) # 64

    # outputs
    # tohle by tu nemuselo byt - vyzkouset
    conv9 = Conv2D(n_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u9)
    outputs = Conv2D(n_classes, 1, padding = 'same')(conv9)
    # outputs = Conv2D(n_classes, 1, padding = "same", activation = "softmax")(u9)

    model = tf.keras.Model(inputs = inputs, outputs = outputs, name = "U-Net")

    return model


# %%
# img_height = 96
# img_width = 128
img_height = IMG_SIZE[0]
img_width = IMG_SIZE[1]
num_channels = 3
n_filters = 32
n_classes = 2

input_size = (img_height, img_width, num_channels)

unet = unet_model((img_height, img_width, num_channels),
                  n_classes = n_classes, n_filters = n_filters)

# %%

unet.summary()

# %%
unet.compile(optimizer = 'adam',
             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
             # loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
             metrics = ['accuracy'])


# %%
EPOCHS = 1
BUFFER_SIZE = 500 # 2000
BATCH_SIZE = 16
# %%

train_dataset = processed_image_train_ds.cache().shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).batch(BATCH_SIZE)
valid_dataset = processed_image_valid_ds.cache().batch(BATCH_SIZE)
test_dataset = processed_image_test_ds.cache().batch(BATCH_SIZE)
#%% Callback

my_callbacks = [
    ModelCheckpoint(folder_to_save_weights + 'stamp.weights.{epoch:04d}.hdf5',
                    save_weights_only=True),
    TensorBoard(log_dir = folder_to_save_results),
    CSVLogger(folder_to_save_results + 'training.log'),
    EarlyStopping(monitor = "val_loss",
                  patience = 10,
                  restore_best_weights = True,
                  verbose = 1)
]

# %%
# model_history = unet.fit(train_dataset, epochs = EPOCHS)

model_history = unet.fit(train_dataset,
                         epochs=EPOCHS,
                         validation_data = valid_dataset,
                         callbacks=[my_callbacks])

# %%

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis = -1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


# %%

plt.plot(model_history.history["accuracy"])
plt.show()


def show_predictions(dataset = None, num = 1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(unet.predict(sample_image[tf.newaxis, ...]))])
#
# # %% Checkpoints
# checkpoints = sorted(glob.glob("./logs/weights/*.hdf5"))
#
# def show_predictions_per_step(dataset = None, checkpoints = None, num = 1):
#     if dataset:
#         for image, mask in dataset.take(num):
#             for checkpoint in checkpoints:
#                 unet.load_weights(checkpoint)
#                 pred_mask = unet.predict(image)
#                 display([image[0], mask[0], create_mask(pred_mask)])
#     else:
#         display([sample_image, sample_mask,
#                  create_mask(unet.predict(sample_image[tf.newaxis, ...]))])
#
#
#
# # %%
# show_predictions(test_dataset, 6)
#
# #%%
#
# show_predictions_per_step(test_dataset, checkpoints, 1)
#
#
# #%% ============================================================================
# #%%  OLD VERSION DISPLAY
# #%%
# #%%
# # %%
# #
# # steps = len(scan_files_test[0:1])
# # checkpoints = sorted(glob.glob("./tmp/unet/160_epoch_2_batchsize_all_data/*.hdf5"))
# # rows = len(checkpoints) + 1
# #
# # # %%
# # plt.figure(figsize=(steps * 10, rows * 10))
# # for i in range(steps):
# #     plt.subplot(rows, steps, i + 1)
# #     plt.imshow(imread(scan_files_test[i]))
# #
# # for i, c in enumerate(checkpoints):
# #     model.load_weights(c)
# #     predicted = model.predict_generator(image_generator(scan_files_test,
# #                                                         randomized=False,
# #                                                         labels=None,
# #                                                         include_weights=False,
# #                                                         batch_size=1,
# #                                                         augment=False),
# #                                         steps=steps)
# #     predicted = np.round(predicted).reshape((steps, IMG_SIZE[0], IMG_SIZE[1]))
# #     for s in range(steps):
# #         plt.subplot(rows, steps, i * steps + s + steps + 1)
# #         plt.imshow(predicted[s])
# #
# # plt.show()
#
#
# # %% Check KB pictures
# # %%
# path_to_kb_data = "../data/kb_invoices/"
#
# scan_kb_files = glob.glob(path_to_kb_data + '*.png')
# scan_kb_files = sorted(scan_kb_files)
#
# # %%
# steps = len(scan_kb_files)
# checkpoints = sorted(glob.glob('./tmp/unet/60_epoch_2_batchsize_all_data/*.hdf5'))
# rows = len(checkpoints) + 1
#
# # %%
# plt.figure(figsize=(steps * 10, rows * 10))
# for i in range(steps):
#     plt.subplot(rows, steps, i + 1)
#     plt.imshow(imread(scan_kb_files[i]))
#
# for i, c in enumerate(checkpoints):
#     model.load_weights(c)
#     predicted = model.predict(image_generator(scan_kb_files,
#                                               randomized=False,
#                                               labels=None,
#                                               include_weights=False,
#                                               batch_size=1,
#                                               augment=False),
#                               steps=steps)
#     predicted = np.round(predicted).reshape((steps, IMG_SIZE[0], IMG_SIZE[1]))
#     for s in range(steps):
#         plt.subplot(rows, steps, i * steps + s + steps + 1)
#         plt.imshow(predicted[s])
#
# plt.show()
