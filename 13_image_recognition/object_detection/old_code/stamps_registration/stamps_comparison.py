# %% LOAD DEPENDECIES
import os
import sys
import time
import glob
import random
import numpy as np
import pandas as pd
import yaml
import pprint
from pathlib import Path
import torch as th
from PIL import Image
from tqdm import tqdm
import PIL.ImageOps
import cv2
import re
import matplotlib.pyplot as plt

# %% AIRLAB PACKAGES
import src.affine_registration as affine_reg
import src.utility as util

# %% PARAMETERS
path_to_dataset = "./data/crops_original_all/signature/"

print(path_to_dataset)

# %% TEST ONE LABEL
imgs = list(sorted(os.listdir(os.path.join(path_to_dataset))))

idx = 0
img_path = os.path.join(path_to_dataset, imgs[idx])
print(img_path)
Image.fromarray(util.image_binarization(img_path)).show()

# %% IMAGE SIMILARITIES ---------

imgs = list(sorted(os.listdir(os.path.join(path_to_dataset))))

fixed_image_path = os.path.join(path_to_dataset, imgs[0])
moving_image_path = os.path.join(path_to_dataset, imgs[1])
# moving_image_path = os.path.join(path_to_dataset, imgs[3])

Image.fromarray(util.image_binarization(fixed_image_path)).show()
Image.fromarray(util.image_binarization(moving_image_path)).show()

device = th.device("cpu")
dtype = th.float32
plot = False
verbose = True
iter_num = 1000



np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


affine_reg.affine(fixed_image_path,
                  moving_image_path,
                  device = th.device("cpu"),
                  dtype = th.float32,
                  plot = True,
                  verbose = True,
                  iter_num = iter_num)

# %% RUN FOR ALL PICTURES AFFINE
stamps_dictionary = pd.DataFrame(columns = ["fixed_image_path",
                                            "moving_image_path",
                                            "loss_sum", "mse_loss", "ncc_loss"])

for i in range(len(imgs)):
# for i in range(0, 2):
    fixed_image_path = os.path.join(path_to_dataset, imgs[i])

    for j in range(i + 1, len(imgs)):
    # for j in range(i + 1, i + 3):
        print(str(i) + ", " + str(j) + " out of " + str(len(imgs)))
        moving_image_path = os.path.join(path_to_dataset, imgs[j])

        image_reg_loss, image_loss_list = affine_reg.affine(fixed_image_path,
                                                            moving_image_path,
                                                            device = th.device("cpu"),
                                                            dtype = th.float32,
                                                            plot = False,
                                                            verbose = False,
                                                            iter_num = 100)

        image_reg_loss = float(image_reg_loss)
        image_loss_list = [float(x) for x in image_loss_list]

        append_df = pd.DataFrame({"fixed_image_path": fixed_image_path,
                                  "moving_image_path": moving_image_path,
                                  "loss_sum": image_reg_loss,
                                  "mse_loss": image_loss_list[0],
                                  "ncc_loss": image_loss_list[1]},
                                 index = [0])

        stamps_dictionary = pd.concat([stamps_dictionary, append_df], ignore_index = True)

stamps_dictionary.to_csv("test_7_NCC.csv", index = False)

# %% LOAD RESULTS ---------

stamps_dictionary = pd.read_csv("test_6.csv")

imgs = list(sorted(os.listdir(os.path.join(path_to_dataset))))

fixed_image_path = os.path.join(path_to_dataset, imgs[1])
moving_image_path = os.path.join(path_to_dataset, imgs[2])

stamps_dictionary_2 = stamps_dictionary.sort_values("loss_sum").reset_index()


idx = 16
moving_image_path = stamps_dictionary_2.iloc[idx].moving_image_path
fixed_image_path = stamps_dictionary_2.iloc[idx].fixed_image_path

affine_reg.affine(fixed_image_path,
                  moving_image_path,
                  device = th.device("cpu"),
                  dtype = th.float32,
                  plot = True,
                  verbose = False,
                  iter_num = 500)

Image.open(stamps_dictionary.iloc[idx]["moving_image_path"]).show()
Image.open(stamps_dictionary.iloc[idx]["fixed_image_path"]).show()


# %% For cycle through all crops

for idx in stamps_dictionary_2.index:
    print(idx)
    moving_image_path = stamps_dictionary_2.iloc[idx].moving_image_path
    fixed_image_path = stamps_dictionary_2.iloc[idx].fixed_image_path
    print(moving_image_path)
    print(fixed_image_path)

    moving_image = util.image_binarization(moving_image_path)
    fixed_image = util.image_binarization(fixed_image_path)

    plt.title('Easy as 1, 2, 3')  # subplot 211 title
    plt.subplot(121)
    plt.imshow(fixed_image, cmap = 'gray')
    plt.title('Fixed Image')

    plt.subplot(122)
    plt.imshow(moving_image, cmap = 'gray')
    plt.title('Moving Image: \n' +
              'SUM: ' + str(stamps_dictionary_2.iloc[idx]["loss_sum"]) + '\n' +
              'MSE: ' + str(stamps_dictionary_2.iloc[idx]["mse_loss"]) + '\n'
              'NCC: ' + str(stamps_dictionary_2.iloc[idx]["ncc_loss"]))

    plt.show()

    input("Press Enter to continue...")



#%% Check Images for binarization

for idx in range(len(imgs)):
    fixed_image_path = os.path.join(path_to_dataset, imgs[idx])

    kernel_size = 8
    binary_image = util.image_binarization_from_so(fixed_image_path, kernel_size = kernel_size)
    original_image = img = cv2.cvtColor(cv2.imread(fixed_image_path), cv2.COLOR_BGR2GRAY)

    binary_image_my = util.image_binarization(fixed_image_path)

    plt.figure(figsize = (20, 20))
    plt.title('Easy as 1, 2, 3')  # subplot 211 title
    plt.subplot(131)
    plt.imshow(original_image, cmap = 'gray')
    plt.title('Original Image')

    plt.subplot(132)
    plt.imshow(binary_image, cmap = 'gray')
    plt.title('binary Image')

    plt.subplot(133)
    plt.imshow(binary_image_my, cmap = 'gray')
    plt.title('My binary Image')

    plt.show()

    input("Press Enter to continue...")

# %% RESEARCH FOR IMAGE REGISTRATION

imgs = list(sorted(os.listdir(os.path.join(path_to_dataset))))

fixed_image_path = os.path.join(path_to_dataset, imgs[0])
moving_image_path = os.path.join(path_to_dataset, imgs[4])
# moving_image_path = os.path.join(path_to_dataset, imgs[3])

Image.fromarray(util.image_binarization(fixed_image_path)).show()
Image.fromarray(util.image_binarization(moving_image_path)).show()

device = th.device("cpu")
dtype = th.float32
plot = False
verbose = False
iter_num = 100


import airlab as al
import time
import imutils

np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


start = time.time()
fixed_image_source = util.image_binarization(fixed_image_path)
fixed_image_source = 255 - fixed_image_source
fixed_image_size = fixed_image_source.shape

moving_image_source = cv2.imread(moving_image_path)
moving_image_source = util.image_binarization(moving_image_path)

for angle in [0, 90, 270, 90]:
    moving_image = imutils.rotate(moving_image_source, angle)

    moving_image = 255 - moving_image
    mov_image_size = moving_image.shape

    IMG_SIZE = (max(fixed_image_size[0], mov_image_size[0]),
                max(fixed_image_size[1], mov_image_size[1]))
    moving_image = cv2.copyMakeBorder(moving_image,
                                      (IMG_SIZE[0] - mov_image_size[0]) // 2,
                                      (IMG_SIZE[0] - mov_image_size[0]) - (
                                                  IMG_SIZE[0] - mov_image_size[0]) // 2,
                                      (IMG_SIZE[1] - mov_image_size[1]) // 2,
                                      (IMG_SIZE[1] - mov_image_size[1]) - (
                                                  IMG_SIZE[1] - mov_image_size[1]) // 2,
                                      cv2.BORDER_CONSTANT, 0)
    fixed_image = cv2.copyMakeBorder(fixed_image_source,
                                     (IMG_SIZE[0] - fixed_image_size[0]) // 2,
                                     (IMG_SIZE[0] - fixed_image_size[0]) - (
                                                 IMG_SIZE[0] - fixed_image_size[0]) // 2,
                                     (IMG_SIZE[1] - fixed_image_size[1]) // 2,
                                     (IMG_SIZE[1] - fixed_image_size[1]) - (
                                                 IMG_SIZE[1] - fixed_image_size[1]) // 2,
                                     cv2.BORDER_CONSTANT, 0)

    moving_image = al.image_from_numpy(moving_image, [1, 1], [0, 0], dtype, device)
    fixed_image = al.image_from_numpy(fixed_image, [1, 1], [0, 0], dtype, device)
    fixed_image, moving_image = al.utils.normalize_images(fixed_image, moving_image)
    registration = al.PairwiseRegistration(verbose = verbose)
    transformation = al.transformation.pairwise.SimilarityTransformation(moving_image, opt_cm = True)
    transformation.init_translation(fixed_image)
    registration.set_transformation(transformation)

    image_loss_1 = al.loss.pairwise.MSE(fixed_image, moving_image)
    image_loss_2 = al.loss.pairwise.NCC(fixed_image, moving_image)
    registration.set_image_loss([image_loss_2])

    optimizer = th.optim.Adam(transformation.parameters(), lr = 0.01, amsgrad = True)
    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(number_of_iterations = iter_num)
    registration.start()
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image, displacement)
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image
    warped_image.image = 1 - warped_image.image

    loss, loss_list = registration._closure()
    loss = float(loss)
    loss_list = [float(x) for x in loss_list]
    end = time.time()
    print("Registration done in:", end - start, "s")
    print("Registration loss:", loss)
    print("=================================================================")
    plt.subplot(131)
    plt.imshow(fixed_image.numpy(), cmap = 'gray')
    plt.title('Fixed Image')

    plt.subplot(132)
    plt.imshow(moving_image.numpy(), cmap = 'gray')
    plt.title('Moving Image')

    plt.subplot(133)
    plt.imshow(warped_image.numpy(), cmap = 'gray')
    plt.title('Warped Moving Image')

    plt.show()


# %% OPENCV REGISTRATION
imgs = list(sorted(os.listdir(os.path.join(path_to_dataset))))

fixed_image_path = os.path.join(path_to_dataset, imgs[0])
moving_image_path = os.path.join(path_to_dataset, imgs[1])
# moving_image_path = os.path.join(path_to_dataset, imgs[3])

affine_reg.image_registration_opencv(fixed_image_path,
                                     moving_image_path,
                                     mode = "orb",
                                     plot_results = True,
                                     plot_keypoints = True)

# %% OPENCV REGISTARTION FOR ALL PICS

start = time.time()

imgs = list(sorted(os.listdir(os.path.join(path_to_dataset))))

stamps_dictionary = pd.DataFrame(columns = ["fixed_image_path",
                                            "moving_image_path",
                                            "ncc_loss"])


for i in range(0, len(imgs)):
# for i in range(0, 2):
    fixed_image_path = os.path.join(path_to_dataset, imgs[i])

    for j in range(i + 1, len(imgs)):
        print(str(i) + " out of " + str(len(imgs)) + ", " + str(j) + " out of " + str(len(imgs)))
        moving_image_path = os.path.join(path_to_dataset, imgs[j])

        image_loss_dict = affine_reg.image_registration_opencv(fixed_image_path,
                                                               moving_image_path,
                                                               mode = "sift",
                                                               plot_results = False,
                                                               plot_keypoints = False)

        append_df = pd.DataFrame({"fixed_image_path": fixed_image_path,
                                  "moving_image_path": moving_image_path,
                                  "ncc_loss": image_loss_dict["ncc_loss"]},
                                 index = [0])

        stamps_dictionary = pd.concat([stamps_dictionary, append_df], ignore_index = True)

end = time.time()

print("Registration done in:", end - start, "s")

stamps_dictionary.to_csv("stamps_dictionary_original_all.csv")


stamps_dictionary_sorted = stamps_dictionary.sort_values("ncc_loss").reset_index()

stamps_dictionary_sorted.head(20)

#%% LOAD RESULTS
stamps_dictionary = pd.read_csv("stamps_dictionary_jpeg_all_2.csv")
stamps_dictionary_sorted = stamps_dictionary.sort_values("ncc_loss").reset_index()

#%%
idx = 2
fixed_image_path = stamps_dictionary_sorted.iloc[idx]["fixed_image_path"]
moving_image_path = stamps_dictionary_sorted.iloc[idx]["moving_image_path"]


affine_reg.image_registration_opencv(fixed_image_path,
                                     moving_image_path,
                                     mode = "sift",
                                     plot_results = True,
                                     plot_keypoints = True)


# %%

for idx in range(10):
    fixed_image_path = stamps_dictionary_sorted.iloc[idx]["fixed_image_path"]
    moving_image_path = stamps_dictionary_sorted.iloc[idx]["moving_image_path"]

    print(fixed_image_path)
    print(moving_image_path)
    affine_reg.image_registration_opencv(fixed_image_path,
                                         moving_image_path,
                                         mode = "sift",
                                         plot_results = False,
                                         plot_keypoints = True,
                                         no_of_keypoints = 20)

    input("Press Enter to continue...")


#%%

stamps_dictionary_names = stamps_dictionary_sorted.__deepcopy__()
stamps_dictionary_names["fixed_image_path"] = stamps_dictionary_names["fixed_image_path"].apply(
    lambda g: g.split("/")[-1])

stamps_dictionary_names["moving_image_path"] = stamps_dictionary_names["moving_image_path"].apply(
    lambda g: g.split("/")[-1])

stamps_dictionary_names = stamps_dictionary_names.drop(columns = ["index", "Unnamed: 0"])
stamps_dictionary_names.to_csv("stamps_jpg_names.csv")