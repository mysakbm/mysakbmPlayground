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

import albumentations as A

# %%
path_to_dataset = "../data/kb_invoices/"
path_to_dataset_augmented = "../data/kb_invoices/single_aug"


# %%

def visualize(image):
    plt.figure(figsize = (10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.show()


image = cv2.imread(path_to_dataset + '20200903_BikeSport_faktura_222846.pdf.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = A.Compose(
    [A.CLAHE(),
     A.RandomRotate90(),
     A.Transpose(),
     A.ShiftScaleRotate(shift_limit = 0.0625, scale_limit = 0.50,
                        rotate_limit = 45, p = .75),
     A.Blur(blur_limit = 3),
     A.OpticalDistortion(),
     A.GridDistortion(),
     A.HueSaturationValue(),
     A.Affine(shear = (-30, 30)),
     A.Perspective(scale = (0.05, 0.1), p = 0.5)])

augmented_image = transform(image = image)['image']
visualize(augmented_image)


