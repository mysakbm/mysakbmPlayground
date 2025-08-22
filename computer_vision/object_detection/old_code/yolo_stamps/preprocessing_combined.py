# %% LOAD DEPENDECIES
import os
# import glob
import random
import numpy as np
import pandas as pd
import yaml
import pprint
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import PIL.ImageOps
import cv2
import re
import matplotlib.pyplot as plt
import src.utility as util

# %% PARAMETERS
path_to_dataset = "../data/signature_datasets/combined/arpita/DateSign"
path_to_dataset_augmented = "../data/stamps_datasets/combined/one_folder_resized_augmented/"

print(path_to_dataset)

SPLIT_DIR = "train/"
SPLIT_DIR = "valid/"
SPLIT_DIR = "test/"

SPLIT_DIRS = ["train/", "valid/", "test/"]

# %% TEST ONE LABEL YOLO

imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, SPLIT_DIR + "/images"))))
labels = list(sorted(os.listdir(os.path.join(path_to_dataset, SPLIT_DIR + "labels"))))

# load yolo default parameters
with open(path_to_dataset + "data.yaml", 'r') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    data_yaml = yaml.load(file, Loader = yaml.FullLoader)

print(data_yaml)

labels_yaml = data_yaml["names"]
labels_yaml = [x.lower() for x in labels_yaml]

idx = 0
img_path = os.path.join(path_to_dataset, SPLIT_DIR + "/images", imgs[idx])
labels_path = os.path.join(path_to_dataset, SPLIT_DIR + "labels", labels[idx])

# %%

util.display_yolo_image(img_path, labels_path, labels_yaml)

# %% PADDING

# load yolo default parameters
with open(path_to_dataset + "data.yaml", 'r') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    data_yaml = yaml.load(file, Loader = yaml.FullLoader)

print(data_yaml)

labels_yaml = data_yaml["names"]
labels_yaml = [x.lower() for x in labels_yaml]

imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, SPLIT_DIR + "/images"))))
labels = list(sorted(os.listdir(os.path.join(path_to_dataset, SPLIT_DIR + "labels"))))

for idx in tqdm(range(len(imgs))):
    img_path = os.path.join(path_to_dataset, SPLIT_DIR + "/images", imgs[idx])
    labels_path = os.path.join(path_to_dataset, SPLIT_DIR + "labels", labels[idx])
    util.resize_images(img_path, labels_path, labels_yaml, IMG_SIZE_NP = (1280, 920, 3))


# %% AUGMENT DATA

def augment_yolo_dataset(path_to_dataset,
                         path_to_dataset_augmented,
                         dataset_type = "test",
                         max_augmentations = 7,
                         crop_max_size = 10,
                         max_rotation_angle = 5,
                         random_flip = 0.4,
                         random_grey = 0.25,
                         random_contrast = 0.3,
                         binarization = 0.25):

    out_images_path = Path(f"{path_to_dataset_augmented}/{dataset_type}/images")
    out_images_path.mkdir(parents = True, exist_ok = True)

    out_labels_path = Path(f"{path_to_dataset_augmented}/{dataset_type}/labels")
    out_labels_path.mkdir(parents = True, exist_ok = True)

    imgs_path = Path(path_to_dataset) / dataset_type / "images"
    masks_path = Path(path_to_dataset) / dataset_type / "labels"

    imgs = list(sorted(os.listdir(imgs_path)))
    masks = list(sorted(os.listdir(masks_path)))

    for idx in tqdm(range(len(imgs))):

        img_path = os.path.join(imgs_path, imgs[idx])
        mask_path = os.path.join(masks_path, masks[idx])

        img = cv2.imread(img_path)
        orig_img_size = img.shape

        # Resize Mask
        blank_image = np.ones(orig_img_size, np.uint8) * 255

        with open(mask_path, "r") as f:
            labels_coords = f.readlines()

        if len(labels_coords) == 0:
            image_name = f"{idx}_{0}.jpeg"
            label_name = f"{idx}_{0}.txt"

            img_out.save(str(out_images_path / image_name), "JPEG")

            with (out_labels_path / label_name).open(mode = "w") as label_file:
                pass

            continue

        for coord_one in labels_coords:
            coord_one = coord_one.rstrip('\n')
            yolo_vals = [float(x) for x in coord_one.split(" ")]
            category_idx = int(yolo_vals[0])

            xmin = int(orig_img_size[1] * (yolo_vals[1] - yolo_vals[3] / 2))
            xmax = int(orig_img_size[1] * (yolo_vals[1] + yolo_vals[3] / 2))

            ymin = int(orig_img_size[0] * (yolo_vals[2] - yolo_vals[4] / 2))
            ymax = int(orig_img_size[0] * (yolo_vals[2] + yolo_vals[4] / 2))

            cv2.rectangle(blank_image,
                          (xmin, ymin),
                          (xmax, ymax),
                          (0, 0, 0),
                          thickness = -1)

        if len(blank_image.shape) == 3:
            blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)

        img = Image.fromarray(img)
        mask = Image.fromarray(255 - np.array(blank_image))

        for jdx in range(max_augmentations + 1):
            if jdx > 0:
                img_out, mask_out = util.augment_data(img, mask,
                                                      crop_max_size = crop_max_size,
                                                      max_rotation_angle = max_rotation_angle,
                                                      random_flip = random_flip,
                                                      random_grey = random_grey,
                                                      random_contrast = random_contrast,
                                                      binarization = binarization)
            else:
                img_out, mask_out = img, mask

            image_name = f"{idx}_{jdx}.jpeg"
            label_name = f"{idx}_{jdx}.txt"

            img_out.save(str(out_images_path / image_name), "JPEG")

            # For debugging
            # canny_output = cv2.Canny(np.array(mask_out), 100, 100 * 2)
            # Image.fromarray(canny_output).show()

            contours, hierarchy = cv2.findContours(np.array(mask_out),
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            with (out_labels_path / label_name).open(mode = "w") as label_file:
                pass

            if len(contours) > 0:
                for one_c in contours:
                    xmin = np.min(one_c[:, :, 0]) / img_out.size[0]
                    xmax = np.max(one_c[:, :, 0]) / img_out.size[0]
                    ymin = np.min(one_c[:, :, 1]) / img_out.size[1]
                    ymax = np.max(one_c[:, :, 1]) / img_out.size[1]

                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    with (out_labels_path / label_name).open(mode = "a") as label_file:
                        label_file.write(
                            f"{category_idx} {xmin + bbox_width / 2} {ymin + bbox_height / 2} "
                            f"{bbox_width} {bbox_height}\n"
                        )


# %%

SPLIT_DIRS = ["train", "valid", "test"]

for d_set in SPLIT_DIRS:
    augment_yolo_dataset(path_to_dataset,
                         path_to_dataset_augmented,
                         dataset_type = d_set,
                         max_augmentations = 15,
                         random_flip = 0.3,
                         crop_max_size = 15,
                         max_rotation_angle = 15,
                         random_grey = 0.25,
                         random_contrast = 0.3,
                         binarization = 0.25)
