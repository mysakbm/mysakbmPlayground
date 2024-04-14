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

# %% PARAMETERS
path_to_dataset = "../data/stamps_datasets/combined/one_folder/"

print(path_to_dataset)

SPLIT_DIR = "train/"

# %% TEST ONE LABEL

imgs = list(sorted(os.listdir(os.path.join(path_to_dataset, SPLIT_DIR + "images"))))
labels = list(sorted(os.listdir(os.path.join(path_to_dataset, SPLIT_DIR + "labels"))))

# load yolo default parameters
with open(path_to_dataset + "data.yaml", 'r') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    data_yaml = yaml.load(file, Loader=yaml.FullLoader)

print(data_yaml)

labels_yaml = data_yaml["names"]
labels_yaml = [x.lower() for x in labels_yaml]

idx = 2
img_path = os.path.join(path_to_dataset, SPLIT_DIR + "images", imgs[idx])
labels_path = os.path.join(path_to_dataset, SPLIT_DIR + "labels", labels[idx])

util.display_yolo_image(img_path, labels_path, labels_yaml)

# %% CHECK DATA YAML AND MODEL YAML

with open(path_to_dataset + '/data.yaml', 'r') as stream:
    try:
        pprint.pprint(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

with open("./src/yolov5/models/" + "/yolov5x.yaml", 'r') as stream:
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

train.run(data = path_to_dataset + "/data.yaml",
          imgsz = 1280,  # 1280
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
