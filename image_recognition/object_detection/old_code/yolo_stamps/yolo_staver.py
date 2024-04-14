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

# IMPORT YOLO PACKAGES
import train
import detect
import val

# %% PARAMETERS
path_to_dataset = "../data/stamps_datasets/combined/one_folder_resized/"

print(path_to_dataset)

SCANS_DIR = "scans/scans/"
TRUTH_DIR = "ground-truth-pixel/ground-truth-pixel/"

CREATE_DATASETS = False
DELETE_EXCESS_FILES = False
DELETE_PICTURES_WITHOUT_MASK = False

# %% DELETE EXCESS FILES # %%
if DELETE_EXCESS_FILES:
    pass

# %% CLEAR PICTURES WITHOUT MASK
if DELETE_PICTURES_WITHOUT_MASK:
    pass

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
imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, "train/images"))))
pixel_mask = list(sorted(os.listdir(os.path.join(path_to_dataset, "train/labels"))))

idx = 1
img_path = os.path.join(path_to_dataset, "train/images", imgs[idx])
pixel_mask_path = os.path.join(path_to_dataset, "train/labels", pixel_mask[idx])
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


# %% CHECK DATA YAML AND MODEL YAML

with open(path_to_dataset + '' + '/data.yaml', 'r') as stream:
    try:
        pprint.pprint(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

with open("./src/yolov5/models/" + "/yolov5l.yaml", 'r') as stream:
    try:
        pprint.pprint(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

# %% SET UP MODEL PARAMETERS FOR CUSTOM YOLO
#
# # load yolo default parameters
# with open(r"./yolov5/models/" + "yolov5l.yaml", 'r') as file:
#     # The FullLoader parameter handles the conversion from YAML
#     # scalar values to Python the dictionary format
#     yolo_model_params_dict = yaml.load(file, Loader=yaml.FullLoader)
#
# print(yolo_model_params_dict)
#
# # set up new parameters
# yolo_model_params_dict["depth_multiple"] = 1
# yolo_model_params_dict["width_multiple"] = 1
#
# print(yolo_model_params_dict)
#
# # write custom yolo parameters
# with open(r"./yolov5/models/" + "custom_stamps_yolov5.yaml", 'w') as file:
#     documents = yaml.dump(yolo_model_params_dict, file)


# %% TRAIN ON DATASET

train.run(data = path_to_dataset + "data.yaml",
          imgsz = 320,  # 1280
          batch_size = 2,  # 5f
          epochs = 1,  # 60
          weights = 'yolov5n.pt',
          cfg = './src/yolov5/models/yolov5n.yaml',
          name = "stamps_yolov5l_results")

# %% PLOT RESULTS
Image.open('./src/yolov5/runs/train/stamps_yolov5l_results/results.png').show()

# %% RUN INFERENCE
detect.run(weights = "./src/yolov5/runs/train/best_trained/weights/best.pt",
           source = path_to_dataset,
           # data = path_to_dataset + "yolo/data.yaml",
           # imgsz = (320, 320),
           imgsz = (1280, 1280),
           # conf_thres = 0.3,
           save_crop = True,
           save_txt = True,
           save_conf = True)

# %%

val.run(data = path_to_dataset + "data.yaml",
        weights = "./src/yolov5/runs/train/best_trained/weights/best.pt",
        batch_size = 2,
        task = "test")