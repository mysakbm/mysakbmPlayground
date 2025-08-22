# %% LOAD DEPENDECIES
import os
# import glob
import random
import numpy as np
import pandas as pd
import yaml
import time
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
path_to_dataset =  "../data/signature_datasets/yolo5tst123_reviewed/"

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
    data_yaml = yaml.load(file, Loader=yaml.FullLoader)

print(data_yaml)

labels_yaml = data_yaml["names"]
labels_yaml = [x.lower() for x in labels_yaml]

idx = 0
img_path = os.path.join(path_to_dataset, SPLIT_DIR + "/images", imgs[idx])
labels_path = os.path.join(path_to_dataset, SPLIT_DIR + "labels", labels[idx])

# %%

util.display_yolo_image(img_path, labels_path, labels_yaml)

# %% Cycle through all images

imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, SPLIT_DIR + "/images"))))
labels = list(sorted(os.listdir(os.path.join(path_to_dataset, SPLIT_DIR + "labels"))))


for idx in range(0, len(imgs)):
    print(str(idx) + " out of " + str(len(imgs)))
    img_path = os.path.join(path_to_dataset, SPLIT_DIR + "/images", imgs[idx])
    labels_path = os.path.join(path_to_dataset, SPLIT_DIR + "labels", labels[idx])

    print(img_path)

    util.display_yolo_image(img_path, labels_path, labels_yaml)

    time.sleep(1.5)
    # input("press enter")

# # Contour Handling
# canny_output = cv2.Canny(255 - pixel_mask, 100, 100 * 2)
# Image.fromarray(canny_output).show()
#
# contours, hierarchy = cv2.findContours(255 - pixel_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# for one_c in contours:
#     # one_c = contours[1]
#     x, y, w, h = cv2.boundingRect(one_c)
#     if w > 5 and h > 10:
#         cv2.rectangle(pixel_mask, (x, y), (x + w, y + h), (155, 155, 155), 1)
#
# Image.fromarray(pixel_mask).show()
#


# %% CHeck number of classes

with open(path_to_dataset + "data.yaml", 'r') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    data_yaml = yaml.load(file, Loader=yaml.FullLoader)

print(data_yaml)


for SPLIT_DIR in SPLIT_DIRS:
    labels = list(sorted(os.listdir(os.path.join(path_to_dataset, SPLIT_DIR + "labels"))))

    number_of_unique_classes = 0
    class_0 = 0
    class_1 = 0
    class_2 = 0

    for idx in range(len(labels)):
        labels_path = os.path.join(path_to_dataset, SPLIT_DIR + "labels", labels[idx])

        with open(labels_path, "r") as f:
            labels_coords = f.readlines()

        labels_yaml_list = [[float(x) for x in coord1.rstrip('\n').split(" ")] for coord1 in
                           labels_coords]

        unique_labels = np.unique([x[0] for x in labels_yaml_list])

        if len(unique_labels) > 0:
            number_of_unique_classes += 1
            # print(labels_path)

        if 0 in unique_labels:
            class_0 += 1

        if 1 in unique_labels:
            class_1 += 1

        if 2 in unique_labels:
            class_2 += 1
            print(labels_path)

    print(number_of_unique_classes)
    print(number_of_unique_classes / len(labels))
    print("class_0: " + str(class_0))
    print("class_1: " + str(class_1))
    print("class_2: " + str(class_2))

#%% CLEARING ONE DATABASE

SPLIT_DIRS = ["train/", "valid/", "test/"]
DELETE_FILES = True
EMPTY_FILES = False

path_to_dataset_folder = "../data/signature_datasets/yolo5tst123/"

print(path_to_dataset_folder)

with open(path_to_dataset_folder + "data.yaml", 'r') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    data_yaml = yaml.load(file, Loader=yaml.FullLoader)

print(data_yaml)

labels_yaml = data_yaml["names"]
labels_yaml = [x for x in labels_yaml]

for SPLIT_DIR in SPLIT_DIRS:
    if not os.path.exists(os.path.join(path_to_dataset_folder, SPLIT_DIR)):
        continue

    imgs = list(sorted(os.listdir(os.path.join(path_to_dataset_folder, SPLIT_DIR + "images"))))
    labels = list(sorted(os.listdir(os.path.join(path_to_dataset_folder, SPLIT_DIR + "labels"))))

    stamp_idx = labels_yaml.index("signature")
    stamp_in = 0
    empty_annot = 0

    for idx in range(len(labels)):
        if not os.path.exists(os.path.join(path_to_dataset_folder, SPLIT_DIR)):
            continue

        img_path = os.path.join(path_to_dataset_folder, SPLIT_DIR + "images", imgs[idx])
        labels_path = os.path.join(path_to_dataset_folder, SPLIT_DIR + "labels", labels[idx])

        with open(labels_path, "r") as f:
            labels_coords = f.readlines()

        labels_yaml_list = [[float(x) for x in coord1.rstrip('\n').split(" ")] for coord1 in
                           labels_coords]

        unique_labels = np.unique([x[0] for x in labels_yaml_list])

        if stamp_idx in unique_labels:
            stamp_in += 1

            if DELETE_FILES:
                with (Path(labels_path)).open(mode = "w") as label_file:
                    pass

                labels_numbers = [int(x[0]) for x in labels_yaml_list]

                for label_row in range(len(labels_numbers)):
                    if stamp_idx == labels_numbers[label_row]:
                        with (Path(labels_path)).open(mode = "a") as label_file:
                            label_file.write(
                                f"{0} {labels_yaml_list[label_row][1]} "
                                f"{labels_yaml_list[label_row][2]} "
                                f"{labels_yaml_list[label_row][3]} "
                                f"{labels_yaml_list[label_row][4]}\n"
                            )
        elif len(labels_yaml_list) == 0:
            # no annotation = empty list of annotations
            empty_annot += 1
            if EMPTY_FILES:
                os.remove(Path(img_path))
                os.remove(Path(labels_path))
        else:
            # print(labels_path)
            if DELETE_FILES:
                with (Path(labels_path)).open(mode = "w") as label_file:
                    pass

                # os.remove(Path(img_path))
                # os.remove(Path(labels_path))

    # print(stamp_in)
    # print(empty_annot)
    print((stamp_in + empty_annot)/ len(labels))

# %% CLEARING ALL CLASSES EXCEPT FOR STAMP FROM ALL DIRS AT ONCE

SPLIT_DIRS = ["train/", "valid/", "test/"]
DELETE_FILES = False
EMPTY_FILES = False

path_to_dataset = "../data/signature_datasets/combined/arpita/"
dataset_to_clean = [x.name for x in os.scandir(path_to_dataset) if x.is_dir()]
# dataset_to_clean.remove("staver")
# dataset_to_clean.remove("one_folder")

for one_dset in dataset_to_clean:
    path_to_dataset_folder = path_to_dataset + one_dset + "/"

    print(path_to_dataset_folder)

    with open(path_to_dataset_folder + "data.yaml", 'r') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        data_yaml = yaml.load(file, Loader=yaml.FullLoader)

    print(data_yaml)

    labels_yaml = data_yaml["names"]
    labels_yaml = [x.lower() for x in labels_yaml]

    for SPLIT_DIR in SPLIT_DIRS:
        if not os.path.exists(os.path.join(path_to_dataset_folder, SPLIT_DIR)):
            continue

        imgs = list(sorted(os.listdir(os.path.join(path_to_dataset_folder, SPLIT_DIR + "images"))))
        labels = list(sorted(os.listdir(os.path.join(path_to_dataset_folder, SPLIT_DIR +
                                                     "labels"))))

        stamp_idx = labels_yaml.index("signature")
        stamp_in = 0
        empty_annot = 0

        for idx in range(len(labels)):
            if not os.path.exists(os.path.join(path_to_dataset_folder, SPLIT_DIR)):
                continue

            img_path = os.path.join(path_to_dataset_folder, SPLIT_DIR + "images", imgs[idx])
            labels_path = os.path.join(path_to_dataset_folder, SPLIT_DIR + "labels", labels[idx])

            with open(labels_path, "r") as f:
                labels_coords = f.readlines()

            labels_yaml_list = [[float(x) for x in coord1.rstrip('\n').split(" ")] for coord1 in
                               labels_coords]

            unique_labels = np.unique([x[0] for x in labels_yaml_list])

            if stamp_idx in unique_labels:
                stamp_in += 1

                if DELETE_FILES:
                    with (Path(labels_path)).open(mode = "w") as label_file:
                        pass

                    labels_numbers = [int(x[0]) for x in labels_yaml_list]

                    for label_row in range(len(labels_numbers)):
                        if stamp_idx == labels_numbers[label_row]:
                            with (Path(labels_path)).open(mode = "a") as label_file:
                                label_file.write(
                                    f"{0} {labels_yaml_list[label_row][1]} "
                                    f"{labels_yaml_list[label_row][2]} "
                                    f"{labels_yaml_list[label_row][3]} "
                                    f"{labels_yaml_list[label_row][4]}\n"
                                )
            elif len(labels_yaml_list) == 0:
                # no annotation = empty list of annotations
                empty_annot += 1
                if EMPTY_FILES:
                    os.remove(Path(img_path))
                    os.remove(Path(labels_path))
            else:
                # print(labels_path)
                if DELETE_FILES:
                    with (Path(labels_path)).open(mode = "w") as label_file:
                        pass

                    # os.remove(Path(img_path))
                    # os.remove(Path(labels_path))

        # print(stamp_in)
        # print(empty_annot)
        print((stamp_in + empty_annot)/ len(labels))