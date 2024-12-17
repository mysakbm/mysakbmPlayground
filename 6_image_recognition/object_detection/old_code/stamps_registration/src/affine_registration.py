# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import cv2
import airlab as al
import src.utility as util

np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


# file_path = "./airlab/examples/affine_registration_2d.py"
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(file_path))))

def affine(fixed_image_path,
           moving_image_path,
           device = th.device("cpu"),
           dtype = th.float32,
           plot = False,
           verbose = False,
           iter_num = 1000):

    start = time.time()

    # set the used data type
    # dtype = th.float32
    # set the device for the computaion to CPU
    # device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the
    # used GPU
    # Here, the GPU with the index 0 is used.
    # device = th.device("cuda:0")

    # load the image data and normalize to [0, 1]

    # Resize of Pictures -------------------------------------------------------
    moving_image = util.image_binarization(moving_image_path)
    fixed_image = util.image_binarization(fixed_image_path)

    # convert intensities so that the object intensities are 1 and the background 0. This is
    # important in order to
    # calculate the center of mass of the object
    fixed_image = 255 - fixed_image
    moving_image = 255 - moving_image

    fixed_image_size = fixed_image.shape
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

    fixed_image = cv2.copyMakeBorder(fixed_image,
                                     (IMG_SIZE[0] - fixed_image_size[0]) // 2,
                                     (IMG_SIZE[0] - fixed_image_size[0]) - (
                                                 IMG_SIZE[0] - fixed_image_size[0]) // 2,
                                     (IMG_SIZE[1] - fixed_image_size[1]) // 2,
                                     (IMG_SIZE[1] - fixed_image_size[1]) - (
                                                 IMG_SIZE[1] - fixed_image_size[1]) // 2,
                                     cv2.BORDER_CONSTANT, 0)

    moving_image = al.image_from_numpy(moving_image, [1, 1], [0, 0], dtype, device)
    fixed_image = al.image_from_numpy(fixed_image, [1, 1], [0, 0], dtype, device)

    # Transform numpy arrays into Airlab format
    fixed_image, moving_image = al.utils.normalize_images(fixed_image, moving_image)

    # UZ PROVEDENO VYSE
    # convert intensities so that the object intensities are 1 and the background 0. This is
    # important in order to
    # calculate the center of mass of the object
    # fixed_image.image = 1 - fixed_image.image
    # moving_image.image = 1 - moving_image.image

    # create pairwise registration object
    registration = al.PairwiseRegistration(verbose = verbose)

    # choose the affine transformation model
    transformation = al.transformation.pairwise.SimilarityTransformation(
        moving_image, opt_cm = True
    )

    # initialize the translation with the center of mass of the fixed image
    transformation.init_translation(fixed_image)

    registration.set_transformation(transformation)

    # choose the Mean Squared Error and NCC as image loss. It seems that it is
    # worth to use both at once
    image_loss_1 = al.loss.pairwise.MSE(fixed_image, moving_image)
    image_loss_2 = al.loss.pairwise.NCC(fixed_image, moving_image)
    registration.set_image_loss([image_loss_1, image_loss_2])

    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(transformation.parameters(), lr = 0.01, amsgrad = True)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(number_of_iterations = iter_num)

    # start the registration
    registration.start()
    # registration.start(EarlyStopping = False)

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image, displacement)

    # set the intensities back to the original for the visualisation
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image
    warped_image.image = 1 - warped_image.image

    end = time.time()

    # get final Loss
    loss, loss_list = registration._closure()

    loss = float(loss)
    loss_list = [float(x) for x in loss_list]

    print("Registration done in:", end - start, "s")
    print("Registration loss:", loss)

    if plot:
        print("=================================================================")

        # print("Result parameters:")
        # transformation.print()

        # plot the results
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

        fig, ax = plt.subplots(figsize = (15, 15))
        ax.set_title("WARPED_affine")
        ax.imshow(warped_image.numpy(), cmap = "gray")

    # # get trasformation matrix
    # matrix = transformation.transformation_matrix.detach().cpu().numpy()
    #
    # A = theta2param(
    #     matrix, moving_image.numpy().shape[1], moving_image.numpy().shape[0]
    # )
    #
    # # square grid
    # square_grid_image = create_square_grid(
    #     warped_image.shape[0], warped_image.shape[1]
    # )
    #
    # square_grid_image = warp_rgb_image(
    #     square_grid_image, displacement, dtype, device
    # )
    #
    # return displacement, A, loss

    return loss, loss_list


def image_registration_opencv(fixed_image_path,
                              moving_image_path,
                              mode: str = "orb",
                              plot_results = True,
                              plot_keypoints = False,
                              no_of_keypoints = 10):

    start = time.time()
    # Open the image files.
    img1_color = cv2.imread(moving_image_path)  # Image to be aligned.
    img2_color = cv2.imread(fixed_image_path)  # Reference image.

    height_1, width_1, _ = img1_color.shape
    height_2, width_2, _ = img2_color.shape

    if ((height_1 > height_2) & (width_1 > width_2)) | \
        ((np.argmax([height_1, width_1, height_2, width_2]) in [0,1])):
        # Convert to grayscale.
        img2 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
        img1 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    else:
        img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    height, width = img2.shape


    if mode.lower() == "orb":
        # Create ORB detector with 5000 features.
        orb_detector = cv2.ORB_create(5000)

        # Find keypoints and descriptors.
        # The first arg is the image, second arg is the mask
        # (which is not required in this case).
        kp1, d1 = orb_detector.detectAndCompute(img1, None)
        kp2, d2 = orb_detector.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        if len(kp2) == 0 or len(kp1) == 0:
            loss_dictionary = {"ncc_loss": None}
            return (loss_dictionary)

    elif mode.lower() == "sift":
        # Create SIFT
        sift = cv2.SIFT_create()
        kp1, d1 = sift.detectAndCompute(img1, None)
        kp2, d2 = sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
    else:
        print("Wrong mode specified")
        return(None)

    # MATCH THE IMAGES
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
    # Sort matches on the basis of their Hamming distance.
    matches = sorted(matches, key = lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    if no_of_matches >= 10:
        # Define empty matrices of shape no_of_matches * 2.
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = kp1[matches[i].queryIdx].pt
            p2[i, :] = kp2[matches[i].trainIdx].pt

        # Find the homography matrix.
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

        if homography is None:
            loss_dictionary = {"ncc_loss": None}
            return loss_dictionary

        # Use this matrix to transform the
        # colored image wrt the reference image.
        transformed_img = cv2.warpPerspective(255 - img1, homography, (width, height))
        loss_dictionary = {"ncc_loss": util.ncc(img2, transformed_img)}
    else:
        loss_dictionary = {"ncc_loss": None}
        return loss_dictionary

    end = time.time()
    print("Registration done in:", end - start, "s")
    print(loss_dictionary)

    if plot_results and no_of_matches >= 10:
        # Save the output.
        plt.subplot(131)
        plt.imshow(img2, cmap = 'gray')
        plt.title('Fixed Image')

        plt.subplot(132)
        plt.imshow(img1, cmap = 'gray')
        plt.title('Moving Image')

        plt.subplot(133)
        plt.imshow(255 - transformed_img, cmap = 'gray')
        plt.title('Warped Moving Image')

        plt.show()

    if plot_keypoints:
        img3 = cv2.drawMatches(img1, kp1,
                              img2, kp2,
                              matches[:no_of_keypoints], None,
                              flags= cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3)
        plt.show()

    return loss_dictionary