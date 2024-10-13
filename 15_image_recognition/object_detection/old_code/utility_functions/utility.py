# %% LOAD DEPENDECIES
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from pathlib import Path
import airlab as al
import random
import PIL.ImageOps
import os
import pandas as pd
import yaml
import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt

# %% AUXILIARY FUNCTION
def calc_sample_weight(image,
                       image_dilate = 0.15,
                       kernel_size = 2):
    # image = rgb2gray(image)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    around_texts = cv2.dilate((image < image_dilate).astype(np.uint8), se)
    class_counts = np.unique(around_texts, return_counts = True)[1]
    class_weight = np.sum(class_counts) / class_counts * np.array([1, 1, 2])[:len(class_counts)]
    class_weight = class_weight / np.max(class_weight)
    weights = np.vectorize(lambda x: class_weight[x])(around_texts)
    return weights, class_counts, class_weight


def image_binarization(img,
                       type = "so"):
    if isinstance(img, str):
        img = cv2.imread(img)

    if type == "my":
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
        image = 1 - img / 255
        weights = calc_sample_weight(image, 0.15, 2)[0]

        # Binarization
        image[weights < 1] = 0
        image[weights >= 1] = 1
        out_binary = 255 - image * 255
    else:
        kernel_size = 8
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        bg = cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
        out_gray = cv2.divide(img, bg, scale = 255)
        out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

    return out_binary


def normalize_channels(img_path):
    img = cv2.imread(img_path)
    # transform to RGB
    img = img[:, :, 0] - (img[:, :, 2] / 3)

    return img

def numpy_image_binarization(numpy_image, kernel_size = 8):
    if len(numpy_image.shape) == 3:
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    bg = cv2.morphologyEx(numpy_image, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(numpy_image, bg, scale = 255)
    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

    return out_binary


def display_yolo_image(img_path, labels_path, labels_yaml):

    img = cv2.imread(img_path)
    img_size = img.shape

    with open(labels_path, "r") as f:
        labels_coords = f.readlines()

    for coord_one in labels_coords:
        coord_one = coord_one.rstrip('\n')
        yolo_vals = [float(x) for x in coord_one.split(" ")]
        category_idx = labels_yaml[int(yolo_vals[0])]

        xmin = int(img_size[1] * (yolo_vals[1] - yolo_vals[3] / 2))
        xmax = int(img_size[1] * (yolo_vals[1] + yolo_vals[3] / 2))

        ymin = int(img_size[0] * (yolo_vals[2] - yolo_vals[4] / 2))
        ymax = int(img_size[0] * (yolo_vals[2] + yolo_vals[4] / 2))

        cv2.rectangle(img,
                      (xmin, ymin),
                      (xmax, ymax),
                      (255, 0, 0),
                      thickness = 2)

        ((label_width, label_height), _) = cv2.getTextSize(
            str(category_idx),
            fontFace = cv2.FONT_HERSHEY_PLAIN,
            # fontScale = 1.75,
            fontScale = 0.75,
            thickness = 1
        )

        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmin + label_width + int(label_width * 0.05),
             ymin - label_height - int(label_height * 0.55)),
            # (int(xmin + label_width), int(xmax - label_height - label_height * 0.55)),
            color = (0, 255, 0),
            thickness = cv2.FILLED
        )

        cv2.putText(
            img,
            str(category_idx),
            org = (xmin, ymin),  # bottom left
            fontFace = cv2.FONT_HERSHEY_PLAIN,
            fontScale = 0.75,
            color = (255, 0, 0),
            # color = (255, 255, 255),
            thickness = 1
        )

    Image.fromarray(img).show()


def resize_images(img_path, labels_path, labels_yaml,
                  IMG_SIZE_NP = (1280, 920, 3)):

    img = cv2.imread(img_path)
    orig_img_size = img.shape

    top_border_size = (IMG_SIZE_NP[0] - orig_img_size[0]) // 2
    bottom_border_size = (IMG_SIZE_NP[0] - orig_img_size[0]) - top_border_size
    left_border_size = (IMG_SIZE_NP[1] - orig_img_size[1]) // 2
    right_border_size = (IMG_SIZE_NP[1] - orig_img_size[1]) - left_border_size

    img_2 = cv2.copyMakeBorder(img,
                               max(0, top_border_size),
                               max(0, bottom_border_size),
                               max(0, left_border_size),
                               max(0, right_border_size),
                               cv2.BORDER_REFLECT)

    img_2 = cv2.resize(img_2,
                       (IMG_SIZE_NP[1], IMG_SIZE_NP[0]),
                       interpolation = cv2.INTER_AREA)

    cv2.imwrite(img_path, img_2)

    # Resize Mask
    blank_image = np.ones(orig_img_size, np.uint8) * 255

    with open(labels_path, "r") as f:
        labels_coords = f.readlines()

    for coord_one in labels_coords:
        coord_one = coord_one.rstrip('\n')
        yolo_vals = [float(x) for x in coord_one.split(" ")]
        category_idx = labels_yaml[int(yolo_vals[0])]

        xmin = int(orig_img_size[1] * (yolo_vals[1] - yolo_vals[3] / 2))
        xmax = int(orig_img_size[1] * (yolo_vals[1] + yolo_vals[3] / 2))

        ymin = int(orig_img_size[0] * (yolo_vals[2] - yolo_vals[4] / 2))
        ymax = int(orig_img_size[0] * (yolo_vals[2] + yolo_vals[4] / 2))

        cv2.rectangle(blank_image,
                      (xmin, ymin),
                      (xmax, ymax),
                      (155, 155, 0),
                      thickness = -1)

        mask_img = cv2.copyMakeBorder(blank_image,
                                      max(0, top_border_size),
                                      max(0, bottom_border_size),
                                      max(0, left_border_size),
                                      max(0, right_border_size),
                                      cv2.BORDER_REFLECT)

        mask_img = cv2.resize(mask_img,
                              (IMG_SIZE_NP[1], IMG_SIZE_NP[0]),
                              interpolation = cv2.INTER_AREA)

        if len(mask_img.shape) == 3:
            mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(255 - mask_img,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Create new file and rewrite the old one
        with (Path(labels_path)).open(mode = "w") as label_file:
            pass

        if len(contours) > 0:
            for one_c in contours:
                xmin = np.min(one_c[:, :, 0]) / img_2.shape[1]
                xmax = np.max(one_c[:, :, 0]) / img_2.shape[1]
                ymin = np.min(one_c[:, :, 1]) / img_2.shape[0]
                ymax = np.max(one_c[:, :, 1]) / img_2.shape[0]

                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                with (Path(labels_path)).open(mode = "a") as label_file:
                    label_file.write(
                        f"{0} {xmin + bbox_width / 2} {ymin + bbox_height / 2} "
                        f"{bbox_width} {bbox_height}\n"
                    )


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
        img_out = numpy_image_binarization(np.array(img_out))
        img_out = Image.fromarray(img_out)

    return (img_out, mask_out)


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


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)

def mse(imageA, imageB):
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err
    
def augment_seg_dataset(path_to_imgs,
                        path_to_dataset_augmented_imgs,
                        path_to_masks,
                        # transform,
                        path_to_dataset_augmented_masks,
                        num_of_iterations = 10,
                        visualize = False):

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

    imgs_names = sorted(os.listdir(path_to_imgs))
    masks_names = sorted(os.listdir(path_to_masks))

    for idx in tqdm(range(len(os.listdir(path_to_imgs)))):
        path_to_img = path_to_imgs + imgs_names[idx]
        path_to_mask = path_to_masks + masks_names[idx]

        image = cv2.imread(path_to_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(path_to_mask)

        for i in range(num_of_iterations):

            transformed_both = transform(image = image, mask = mask)
            transformed_image = transformed_both['image']
            transformed_mask = transformed_both['mask']

            if visualize:
                visualize(transformed_image, transformed_mask)

            cv2.imwrite(path_to_dataset_augmented_imgs +
                        imgs_names[idx].split(".")[0] + "_" + str(i) + ".jpg",
                        transformed_image)

            cv2.imwrite(path_to_dataset_augmented_masks +
                        masks_names[idx].split(".")[0] + "_" + str(i) + ".jpg",
                        transformed_mask)
