# %% LOAD DEPENDECIES
import os
# import glob
import random
import numpy as np
import pandas as pd
import yaml
import pprint
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm
import PIL.ImageOps
import cv2
import re
import matplotlib.pyplot as plt
import src.utility as util

# # IMPORT YOLO PACKAGES
# import train
# import detect

# %% PARAMETERS
path_to_dataset = "../data/stamps_datasets/staver/data/"

print(path_to_dataset)

SCANS_DIR = "scans/scans/"
TRUTH_DIR = "ground-truth-pixel/ground-truth-pixel/"
MAPS_DIR = "ground-truth-maps/ground-truth-maps"

CREATE_DATASETS = False
DELETE_EXCESS_FILES = False
DELETE_PICTURES_WITHOUT_MASK = False

# %% DELETE EXCESS FILES # %%
if DELETE_EXCESS_FILES:
    exces_scans = list(sorted(os.listdir(os.path.join(path_to_dataset, "scans/scans"))))
    del_files = [os.path.join(path_to_dataset, "scans/scans/") + s for s in exces_scans if
                 re.search(r'(.*?4[0-9][0-9].*?)', s)]
    del_files = del_files[1:]

    [os.remove(file) for file in del_files]

# %% CHECKING IMAGES

imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, SCANS_DIR))))
pixel_masks = list(sorted(os.listdir(os.path.join(path_to_dataset, TRUTH_DIR))))
masks = list(sorted(os.listdir(os.path.join(path_to_dataset, MAPS_DIR))))

for idx in range(8, len(imgs)):

    img_path = os.path.join(path_to_dataset, SCANS_DIR, imgs[idx])
    mask_path = os.path.join(path_to_dataset, MAPS_DIR, masks[idx])
    pixel_mask_path = os.path.join(path_to_dataset, TRUTH_DIR, pixel_masks[idx])

    print(img_path)
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    pixel_mask = cv2.imread(pixel_mask_path)

    plt.figure(figsize = (20, 20))
    plt.subplot(131)
    plt.imshow(img, cmap = 'gray')
    plt.title('img')

    plt.subplot(132)
    plt.imshow(mask, cmap = 'gray')
    plt.title('mask')

    plt.subplot(133)
    plt.imshow(pixel_mask, cmap = 'gray')
    plt.title('pixel_mask')

    plt.show()

    input("Wait")
# %% TEST ONE LABEL

imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, "scans/scans"))))
pixel_mask = list(
    sorted(os.listdir(os.path.join(path_to_dataset, "ground-truth-pixel/ground-truth-pixel"))))

idx = 0
img_path = os.path.join(path_to_dataset, "scans/scans", imgs[idx])
pixel_mask_path = os.path.join(path_to_dataset, "ground-truth-pixel/ground-truth-pixel",
                               pixel_mask[idx])

img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
mask = Image.open(pixel_mask_path).convert("L")
mask = 255 - np.array(mask)

category_idx = 0

pos = np.where(mask)
xmin = np.min(pos[1])
xmax = np.max(pos[1])
ymin = np.min(pos[0])
ymax = np.max(pos[0])

cv2.rectangle(
    img,
    (int(xmin), int(ymin)),
    (int(xmax), int(ymax)),
    color = (0, 255, 0),
    thickness = 2
)

((label_width, label_height), _) = cv2.getTextSize(
    "label",
    fontFace = cv2.FONT_HERSHEY_PLAIN,
    fontScale = 1.75,
    thickness = 2
)

cv2.rectangle(
    img,
    (int(xmin), int(ymin)),
    (int(xmin + label_width + label_width * 0.05), int(ymin + label_height + label_height * 0.25)),
    color = (0, 255, 0),
    thickness = cv2.FILLED
)

cv2.putText(
    img,
    "label",
    org = (int(xmin), int(ymin + label_height + label_height * 0.25)),  # bottom left
    fontFace = cv2.FONT_HERSHEY_PLAIN,
    fontScale = 1.75,
    color = (255, 255, 255),
    thickness = 2
)

Image.fromarray(img).show()

# %% TEST ONE YOLO AUGMENTED ===================================================
imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, "yolo/train/images"))))
pixel_mask = list(sorted(os.listdir(os.path.join(path_to_dataset, "yolo/train/labels"))))

idx = 1
img_path = os.path.join(path_to_dataset, "yolo/train/images", imgs[idx])
pixel_mask_path = os.path.join(path_to_dataset, "yolo/train/labels", pixel_mask[idx])
print(img_path)

img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
with open(pixel_mask_path) as f:
    lines = f.readline()

lines[:-2].split(" ")

category_idx = 0

xmin = img.shape[1] * (float(lines[:-2].split(" ")[1]) - float(lines[:-2].split(" ")[3]) / 2)
xmax = img.shape[1] * (float(lines[:-2].split(" ")[1]) + float(lines[:-2].split(" ")[3]) / 2)
ymin = img.shape[0] * (float(lines[:-2].split(" ")[2]) - float(lines[:-2].split(" ")[4]) / 2)
ymax = img.shape[0] * (float(lines[:-2].split(" ")[2]) + float(lines[:-2].split(" ")[4]) / 2)

cv2.rectangle(
    img,
    (int(xmin), int(ymin)),
    (int(xmax), int(ymax)),
    color = (0, 255, 0),
    thickness = 2
)

((label_width, label_height), _) = cv2.getTextSize(
    "label",
    fontFace = cv2.FONT_HERSHEY_PLAIN,
    fontScale = 1.75,
    thickness = 2
)

cv2.rectangle(
    img,
    (int(xmin), int(ymin)),
    (int(xmin + label_width + label_width * 0.05), int(ymin + label_height + label_height * 0.25)),
    color = (0, 255, 0),
    thickness = cv2.FILLED
)

cv2.putText(
    img,
    "label",
    org = (int(xmin), int(ymin + label_height + label_height * 0.25)),  # bottom left
    fontFace = cv2.FONT_HERSHEY_PLAIN,
    fontScale = 1.75,
    color = (255, 255, 255),
    thickness = 2
)

Image.fromarray(img).show()


# %% CREATE DATA SETS

def create_dataset(indices, dataset_type,
                   max_augmentations = 7,
                   crop_max_size = 10,
                   max_rotation_angle = 5,
                   random_flip = 0.4,
                   random_grey = 0.25,
                   random_contrast = 0.3,
                   binarization = 0.25):

    imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, SCANS_DIR))))
    pixel_masks = list(sorted(os.listdir(os.path.join(path_to_dataset, TRUTH_DIR))))
    masks = list(sorted(os.listdir(os.path.join(path_to_dataset, MAPS_DIR))))

    images_path = Path(f"{path_to_dataset}/yolo/{dataset_type}/images")
    images_path.mkdir(parents = True, exist_ok = True)

    labels_path = Path(f"{path_to_dataset}/yolo/{dataset_type}/labels")
    labels_path.mkdir(parents = True, exist_ok = True)

    for idx in tqdm(indices):
        category_idx = 0
        img_path = os.path.join(path_to_dataset, SCANS_DIR, imgs[idx])
        mask_path = os.path.join(path_to_dataset, MAPS_DIR, masks[idx])
        # pixel_mask_path = os.path.join(path_to_dataset, TRUTH_DIR, pixel_masks[idx])

        img = cv2.imread(img_path)
        mask = util.image_binarization(mask_path)
        # pixel_mask = util.image_binarization(pixel_mask_path)

        img = Image.fromarray(img)
        mask = Image.fromarray(255 - np.array(mask))

        for jdx in range(max_augmentations + 1):
            if jdx > 0:
                img_out, mask_out = augment_data(img, mask,
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

            img_out.save(str(images_path / image_name), "JPEG")

            # For debugging
            # canny_output = cv2.Canny(np.array(mask_out), 100, 100 * 2)
            # Image.fromarray(canny_output).show()

            contours, hierarchy = cv2.findContours(np.array(mask_out),
                                                   cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                for one_c in contours:
                    xmin = np.min(one_c[:, :, 0]) / img_out.size[0]
                    xmax = np.max(one_c[:, :, 0]) / img_out.size[0]
                    ymin = np.min(one_c[:, :, 1]) / img_out.size[1]
                    ymax = np.max(one_c[:, :, 1]) / img_out.size[1]

                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    with (labels_path / label_name).open(mode = "a") as label_file:
                        label_file.write(
                            f"{category_idx} {xmin + bbox_width / 2} {ymin + bbox_height / 2} "
                            f"{bbox_width} {bbox_height}\n"
                        )
            else:
                with (labels_path / label_name).open(mode = "a") as label_file:
                    pass


# %% AUGMENT DATA
def augment_data(img, mask,
                 crop_max_size = 50,
                 max_rotation_angle = 5,
                 random_flip = 0.4,
                 random_grey = 0.25,
                 random_contrast = 0.3,
                 binarization = 0.
                 ):

    img_sizes = img.size

    # Zoom: 0 % Minimum Zoom, 10 % Maximum Zoom
    random_x = random.randint(0, crop_max_size)
    random_y = random.randint(0, crop_max_size)

    random_x_2 = random.randint(0, crop_max_size)
    random_y_2 = random.randint(0, crop_max_size)

    img_out = img. \
        crop([random_x,
              random_y,
              img_sizes[0] - random_x_2,
              img_sizes[1] - random_y_2]). \
        resize(img_sizes)

    mask_out = mask. \
        crop([random_x,
              random_y,
              img_sizes[0] - random_x_2,
              img_sizes[1] - random_y_2]). \
        resize(img_sizes, PIL.Image.Resampling.NEAREST)

    # Rotation: Between - 5° and +5°
    rot_angle = random.uniform(-max_rotation_angle, max_rotation_angle)
    img_out = img_out.rotate(rot_angle, fillcolor = "white")
    mask_out = mask_out.rotate(rot_angle)

    # Flip
    if random_flip > 0 and random_flip >= random.random():
        img_out = img_out.transpose(method = Image.Transpose.FLIP_LEFT_RIGHT)
        mask_out = mask_out.transpose(method = Image.Transpose.FLIP_LEFT_RIGHT)

    # Grey
    if random_grey > 0 and random_grey >= random.random():
        img_out = img_out.convert("L")

    # Contrast
    if random_contrast > 0 and random_contrast >= random.random():
        enhancer = ImageEnhance.Contrast(img_out)
        factor = random.uniform(0.5, 3)
        img_out = enhancer.enhance(factor)

    if binarization > 0 and binarization >= random.random():
        img_out = util.numpy_image_binarization(np.array(img_out))
        img_out = Image.fromarray(img_out)

    return (img_out, mask_out)


# %% CREATE DATASETS

if CREATE_DATASETS:
    imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, "scans/scans"))))

    indices = list(range(len(imgs)))
    random.shuffle(indices)

    # Tohle poustet jen pri pretrenovani a je treba smazat predchozi data!

    create_dataset(indices, 'train',
                   max_augmentations = 15,
                   random_flip = 0.3,
                   crop_max_size = 15,
                   max_rotation_angle = 15,
                   random_grey =  0.25,
                   random_contrast =  0.3,
                   binarization =  0.25)

    create_dataset(indices[-60:], 'valid',
                   max_augmentations = 20,
                   random_flip = 0.5,
                   crop_max_size = 15,
                   max_rotation_angle = 20,
                   random_grey =  0.4,
                   random_contrast = 0.4,
                   binarization =  0.4)

    # For test purposes
    create_dataset(indices[0:2], 'test',
                   max_augmentations = 10,
                   random_flip = 0.3,
                   crop_max_size = 15,
                   max_rotation_angle = 10,
                   random_grey = True,
                   random_contrast = True,
                   binarization = True)

    data_yaml_dict = {"train": os.path.abspath(path_to_dataset) + "/yolo/train/images",
                      "val": os.path.abspath(path_to_dataset) + "/yolo/valid/images",
                      "nc": 1,
                      "names": ["Stamp"]}


    with open(path_to_dataset + 'yolo/data.yaml', 'w') as file:
        documents = yaml.dump(data_yaml_dict, file)

# %% CHECK DATA YAML AND MODEL YAML

# with open("./Staver-2/" + "/data.yaml", 'r') as stream:
#     try:
#         pprint.pprint(yaml.safe_load(stream))
#     except yaml.YAMLError as exc:f
#         print(exc)

with open(path_to_dataset + 'yolo/' + '/data.yaml', 'r') as stream:
    try:
        pprint.pprint(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

with open("./src/yolov5/models/" + "/yolov5l.yaml", 'r') as stream:
    try:
        pprint.pprint(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

# Kolik je vlastne prazdnych obrazku bez masky?
masks = list(sorted(os.listdir(os.path.join(path_to_dataset, MAPS_DIR))))
empty_pics = [x for x in masks if len(np.unique(util.image_binarization(os.path.join(
    path_to_dataset,
    MAPS_DIR, x)))) == 1]

len(empty_pics)
len(empty_pics) / len(list(sorted(os.listdir(os.path.join(path_to_dataset, "scans/scans")))))
