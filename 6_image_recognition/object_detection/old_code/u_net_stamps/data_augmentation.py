# %% LOAD DEPENDECIES
import os
# import glob
# import random
import numpy as np
# import pandas as pd
# import yaml
# import pprint
# from pathlib import Path
# from PIL import Image
from tqdm import tqdm
# import PIL.ImageOps
import cv2
# import re
import matplotlib.pyplot as plt

import albumentations as A

# %%
path_to_dataset = "../data/stamps_datasets/staver/data/"
path_to_dataset_augmented_imgs = "../data/stamps_datasets/staver/data/augmented/imgs/"
path_to_dataset_augmented_masks = "../data/stamps_datasets/staver/data/augmented/masks/"

path_to_imgs = path_to_dataset + "/scans/scans/"
path_to_masks = path_to_dataset + "/ground-truth-pixel/ground-truth-pixel/"

# %% MAKE DIR

os.makedirs(path_to_dataset_augmented_imgs, exist_ok = True)
os.makedirs(path_to_dataset_augmented_masks, exist_ok = True)


# %%

def visualize(image,
              mask = None):

    plt.figure(figsize = (15, 15))
    plt.axis('off')

    if mask is not None:
        plt.subplot(121)
        plt.imshow(image, cmap = 'gray')
        plt.title('Image')

        plt.subplot(122)
        plt.imshow(mask, cmap = 'gray')
        plt.title('Mask')

        plt.show()
    else:
        plt.imshow(image)
        plt.show()


# %%

transform = A.Compose(
    [A.CLAHE(),
     A.RandomRotate90(),
     A.Transpose(),
     A.ShiftScaleRotate(shift_limit = 0.0625, scale_limit = 0.50,
                        rotate_limit = 40, p = .65),
     A.Blur(blur_limit = 3),
     A.OpticalDistortion(),
     A.GridDistortion(),
     A.HueSaturationValue(),
     A.Affine(shear = (-15, 15)),
     A.Perspective(scale = (0.05, 0.1), p = 0.5)])

# %%

idx = 1
path_to_img = path_to_imgs + sorted(os.listdir(path_to_imgs))[idx]
path_to_mask = path_to_masks + sorted(os.listdir(path_to_masks))[idx]

image = cv2.imread(path_to_img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(path_to_mask)

# %%

transformed_both = transform(image = image, mask = 255 - mask)
transformed_image = transformed_both['image']
transformed_mask = transformed_both['mask']
visualize(transformed_image, transformed_mask)

# %% AUGMENT DATA

num_of_iterations = 10

imgs_names = sorted(os.listdir(path_to_imgs))
masks_names = sorted(os.listdir(path_to_masks))

for idx in tqdm(range(len(os.listdir(path_to_imgs)))):
    path_to_img = path_to_imgs + imgs_names[idx]
    path_to_mask = path_to_masks + masks_names[idx]

    image = cv2.imread(path_to_img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(path_to_mask)

    for i in range(num_of_iterations):

        transformed_both = transform(image = image, mask = 255 - mask)
        transformed_image = transformed_both['image']
        transformed_mask = transformed_both['mask']
        # visualize(transformed_image, transformed_mask)

        cv2.imwrite(path_to_dataset_augmented_imgs +
                    imgs_names[idx].split(".")[0] + "_" + str(i) + ".png",
                    transformed_image)

        cv2.imwrite(path_to_dataset_augmented_masks +
                    masks_names[idx].split(".")[0] + "_" + str(i) + ".png",
                    transformed_mask)

# %%




