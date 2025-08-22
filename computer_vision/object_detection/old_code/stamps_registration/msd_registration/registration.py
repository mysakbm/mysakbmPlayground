# LOAD DEPENDENCIES ============================================================
# %load_ext autoreload
# %autoreload 2
import itertools
import json
import os
from datetime import date
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from PIL import Image

import src.affine_registration
import src.elastic_registration
from src.utils import create_square_grid, renormalize, warp_rgb_image

np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

# SET GPU ======================================================================
# gpu if available
gpu_id = 7
if th.cuda.is_available():
    device = th.device("cuda:{}".format(str(gpu_id)))
else:
    device = th.device("cpu")
# set the used data type
dtype = th.float32


# AUX FUNCTIONS ================================================================


def stack_col_of_ones(arr):
    return np.hstack((arr, np.ones((len(arr), 1))))


def create_image_matching(
        moving_idx: "list(int)",
        fixed_idx: "list(int)",
        matching_type: str = "pairs",
):
    """Create image matching.

    Returns:
        list(dict): list({'moving': xx , 'fixed': xx})

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.

            moving_idx = [1,2,3]
            fixed_idx = [5,6,7]
            print(create_image_matching(moving_idx, fixed_idx, 'pairs'))

            >> [{'moving': 1, 'fixed': 5}, {'moving': 2, 'fixed': 6}, {'moving': 3, 'fixed': 7}]

            moving_idx = [1,2,3]
            fixed_idx = [5,6]
            print(create_image_matching(moving_idx, fixed_idx, 'template'))

            >> [{'moving': 1, 'fixed': 5}, {'moving': 1, 'fixed': 6}, {'moving': 2, 'fixed': 5}, {'moving': 2, 'fixed': 6}, {'moving': 3, 'fixed': 5}, {'moving': 3, 'fixed': 6}]

    """
    if matching_type == "pairs":
        image_matching = [
            {"moving": x[0], "fixed": x[1]} for x in zip(moving_idx, fixed_idx)
        ]
    if matching_type == "template":
        image_matching = [
            {"moving": x[0], "fixed": x[1]}
            for x in itertools.product(moving_idx, fixed_idx)
        ]
    return image_matching


############################################################################
# SET INPUTS
############################################################################
# set dir to project root, we can use relative paths
if "USER" in os.environ:
    os.chdir(
        "/SFS/user/ry/{}/2020-digitalpathologyds".format(os.environ["USER"])
    )
    # base_path_input = Path("data/")

    cut_brains_input = Path("brains_cut/")
    atlas_input = Path("atlas_allen_complete_1400_900")

    base_path_output = Path("runs/")
else:
    base_path_input = Path(
        os.environ["USERPROFILE"]
        + "\\"
        + "Merck Sharp & Dohme, Corp"
        + "\\"
        + "IT Global Data Science - 2020-DigitalPathologyDS-MRL"
        + "\\"
        + "data"
        + "\\"
    )
    base_path_output = Path("runs/")

# change path where to store results
# path_results = Path(base_path_output) / "brains_cut_to_atlas"/"STR_21"#str(date.today()) /
# "DPD7_Renee_alternative"
# path_results = Path(base_path_output) / "my_test_registration"
path_results = Path(base_path_output) / "brains_cut_to_atlas" / "STR_23"

# image channel
# 0 - Red channel
# 1 - Green channel
# 2 - Blue channel
# 3 - Red - Green/3
# 4 - all 3 channels
channel = {"moving": 0, "fixed": 2}
# moving_folder = "2174"
# fixed_folder = "atlas_allen_complete"

moving_folder = "r41b21"
fixed_folder = "atlas_allen_complete_1400_900"
# image matching for registration
matching_type = "pairs"
# moving_idx = [3, 4, 6, 9, 11, 13, 15, 17, 1, 2, 5, 8, 10, 12, 14, 16, 18]
# fixed_idx = [81, 82, 83, 83, 82, 80, 82, 81, 79, 83, 84, 82, 83, 79, 83, 83, 82]
moving_idx = [1, 6, 17, 14, 2, 19, 18, 7, 3, 20, 11, 8, 4, 15, 12, 9, 5, 16, 13, 10]
fixed_idx = [47, 50, 61, 53, 53, 55, 53, 52, 52, 51, 50, 53, 49, 54, 56, 54, 50, 51, 50, 49]

# draw only subset of annotations, set None to draw all available
subset_moving_annotations = None
subset_fixed_annotations = None

# REGISTRATION PARAMETERS
affine_registration_method = "affine"

# 'diffeomorphic_bspline' or 'demons'
elastic_registration_method = "diffeomorphic_bspline"

# plot results
plot_results = False

############################################
# only check, not neccessary to change paths
# moving
# annotation mapping table
moving_pairing = pd.read_csv(
    # base_path_input / moving_folder / "pairing.csv",
    cut_brains_input / moving_folder / "pairing.csv",
    index_col = 0,
    na_values = [None],
    keep_default_na = False,
)

# images
# moving_images = Path(base_path_input / moving_folder / "images_fixed")
moving_images = Path(cut_brains_input / moving_folder / "images")

# annotations
# moving_annotations = Path(base_path_input / moving_folder / "annotations")
moving_annotations = Path(cut_brains_input / moving_folder / "annotations")

# fixed
# annotation mapping table
fixed_pairing = pd.read_csv(
    # base_path_input / fixed_folder / "pairing.csv",
    atlas_input / "pairing.csv",
    index_col = 0,
    na_values = [None],
    keep_default_na = False,
)

# images
# fixed_images = Path(base_path_input / fixed_folder / "images_fixed")
fixed_images = Path(atlas_input / "images_fixed")

# annotations
# fixed_annotations = Path(base_path_input / fixed_folder / "annotations")
fixed_annotations = Path(atlas_input / "annotations")

# results
# where to save affine registered images
path_affine = Path(path_results / "affine" / "moving")
# where to save elastic registered images
path_elastic = Path(path_results / "elastic" / "moving")
# where to save fixed to moving affine registered images
path_affine_fixed = Path(path_results / "affine" / "fixed")
# where to save fixed to moving elastic registered images
path_elastic_fixed = Path(path_results / "elastic" / "fixed")

# create config
config = dict(
    (name, eval(name))
    for idx, name in enumerate(
        [
            "channel",
            "moving_folder",
            "fixed_folder",
            "matching_type",
            "affine_registration_method",
            "elastic_registration_method",
            "subset_moving_annotations",
            "subset_fixed_annotations",
        ]
    )
)
# save config to csv
dest_config = path_results / "info.json"
dest_config.parent.mkdir(parents = True, exist_ok = True)
with open(dest_config, "w") as fp:
    json.dump(config, fp, indent = 2)

################################################################################
# Match images for registration
################################################################################

image_matching = create_image_matching(moving_idx, fixed_idx, matching_type)

################################################################################
# REGISTRATION LOOP
################################################################################

# placeholder for losses
losses = []
# RUN LOOP
for idx in image_matching:
    f = (
        list(moving_pairing.loc[idx["moving"]])[0],
        list(fixed_pairing.loc[idx["fixed"]])[0],
    )

    f_name = (os.path.splitext(f[0])[0], os.path.splitext(f[1])[0])

    # moving_path
    moving = str(moving_images / f[0])

    # fixed_pathf[1]
    fixed = str(fixed_images / f[1])

    # get ofset values for image (coordinates of top left corner)
    # used to shift annotation coordinates
    # TODO make more human readable, author: Martin Vagenknecht

    # FIXME ofset_moving reversed
    test_path_moving = os.path.splitext(f[0])[0].split("xx")[1].split("_")
    test_path_moving.reverse()
    ofset_moving = list(
        map(int, test_path_moving)
    )
    test_path_fixed = os.path.splitext(f[1])[0].split("xx")[1].split("_")
    test_path_fixed.reverse()
    ofset_fixed = list(
        map(int, test_path_fixed)
    )

    """
    ofset_moving = list(
        map(int, os.path.splitext(f[0])[0].split("xx")[1].split("_"))
    )

    ofset_fixed = list(
        map(int, os.path.splitext(f[1])[0].split("xx")[1].split("_"))
    )"""

    # check if image size matches, otherwise skip
    if not cv2.imread(fixed, -1).shape == cv2.imread(moving, -1).shape:
        print(
            "================================================================="
        )
        print((f, idx))
        print("failed")
        print("fixed: {}".format(cv2.imread(fixed, -1).shape))
        print("moving: {}".format(cv2.imread(moving, -1).shape))
        continue
    # print which images are currently evaluated
    print("=================================================================")
    print((f, idx))

    #####################
    # AFFINE REGISTRATION
    # run affine registration
    print("Affine")
    affine_registration = getattr(
        src.affine_registration, affine_registration_method
    )

    displacement, A, loss_affine = affine_registration(
        fixed, moving, channel, device, dtype, plot = plot_results
    )

    # warped
    warped_image = np.array(Image.open(moving))
    warped_image = warp_rgb_image(warped_image, displacement, dtype, device)
    warped = path_affine / f[0]
    warped.parent.mkdir(parents = True, exist_ok = True)
    Image.fromarray(warped_image).save(warped)

    # moving
    moving_image = np.array(Image.open(moving))

    # fixed
    fixed_image = np.array(Image.open(fixed))

    # make gifs
    im = Image.fromarray(moving_image)
    im2 = Image.fromarray(fixed_image)

    dest = (
            path_results
            / "results"
            / "moving_vs_fixed"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".gif")
    )

    dest.parent.mkdir(parents = True, exist_ok = True)

    im.save(dest, save_all = True, append_images = [im2], duration = 700, loop = 0)

    # transform annotations
    ann_folder = list(moving_pairing.loc[idx["moving"]])[1]
    if ann_folder != "":
        for ann in os.listdir(moving_annotations / ann_folder):
            an = pd.read_csv(moving_annotations / ann_folder / ann)
            an_np = np.array(an)
            an = an_np - ofset_moving
            # transform coordinates
            an_ = stack_col_of_ones(an).transpose()
            an_warped = A @ an_
            an_warped = np.round(
                an_warped.astype(np.int32).transpose(), decimals = 0
            )
            # save warped annotation
            dest = path_affine / "annotations" / ann_folder / ann
            dest.parent.mkdir(parents = True, exist_ok = True)
            pd.DataFrame(an_warped).to_csv(dest, index = False)

    ######################
    # ELASTIC REGISTRATION

    # moving_path changes to affine transform
    moving = str(path_affine / f[0])

    # run elastic registration
    print("Elastic")
    elastic_registration = getattr(
        src.elastic_registration, elastic_registration_method
    )

    (
        displacement,
        displacement_full_coords,
        inverse_displacement,
        shape,
        loss_elastic,
    ) = elastic_registration(
        fixed, moving, channel, device, dtype, plot = plot_results
    )

    # save loses
    losses.append(
        {
            "moving_pairing_id": idx["moving"],
            "moving": f[0],
            "fixed_pairing_id": idx["fixed"],
            "fixed": f[1],
            "loss": loss_elastic,
        }
    )

    # transform annotations
    ann_folder = list(moving_pairing.loc[idx["moving"]])[1]
    if ann_folder != "":
        for ann in os.listdir(path_affine / "annotations" / ann_folder):
            an = pd.read_csv(path_affine / "annotations" / ann_folder / ann)
            an = np.array(an)
            an_tr = []
            for p in an:
                d = inverse_displacement[0][p[1]][p[0]]
                x = int(renormalize(d[0], (-1, 1), (0, shape[1])))
                y = int(renormalize(d[1], (-1, 1), (0, shape[0])))
                an_tr.append([x, y])
            dest = path_elastic / "annotations" / ann_folder / ann
            dest.parent.mkdir(parents = True, exist_ok = True)
            pd.DataFrame(an_tr).to_csv(dest, index = False)

    # warped
    warped_image = np.array(Image.open(moving))
    warped_image = warp_rgb_image(warped_image, displacement, dtype, device)
    warped = path_elastic / f[0]
    warped.parent.mkdir(parents = True, exist_ok = True)
    Image.fromarray(warped_image).save(warped)

    # square grid
    square_grid_image = create_square_grid(
        warped_image.shape[0], warped_image.shape[1]
    )

    square_grid_image = warp_rgb_image(
        square_grid_image, displacement, dtype, device
    )

    im = Image.fromarray(square_grid_image)

    dest = (
            path_results
            / "results"
            / "square_grid_warped(a+e)"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".gif")
    )

    dest.parent.mkdir(parents = True, exist_ok = True)
    im.save(dest)
    # moving
    moving_image = np.array(Image.open(moving))
    # fixed
    fixed_image = np.array(Image.open(fixed))
    # make gifs
    im = Image.fromarray(warped_image)
    im2 = Image.fromarray(fixed_image)
    dest = (
            path_results
            / "results"
            / "warped(a+e)_vs_fixed"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".gif")
    )
    dest.parent.mkdir(parents = True, exist_ok = True)
    im.save(dest, save_all = True, append_images = [im2], duration = 700, loop = 0)
    im = Image.fromarray(moving_image)
    im2 = Image.fromarray(warped_image)
    dest = (
            path_results
            / "results"
            / "warped(a)_vs_warped(a+e)"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".gif")
    )
    dest.parent.mkdir(parents = True, exist_ok = True)
    im.save(dest, save_all = True, append_images = [im2], duration = 700, loop = 0)
    # transform annotations backward fixed to moving
    # elastic fixed to moving
    ann_folder = list(fixed_pairing.loc[idx["fixed"]])[1]
    if ann_folder != "":
        for ann in os.listdir(fixed_annotations / ann_folder):
            an = pd.read_csv(fixed_annotations / ann_folder / ann)
            an = np.array(an)
            an = an - ofset_fixed
            an_tr = []
            for p in an:
                d = displacement_full_coords[0][p[1]][p[0]]
                x = int(renormalize(d[0], (-1, 1), (0, shape[1])))
                y = int(renormalize(d[1], (-1, 1), (0, shape[0])))
                an_tr.append([x, y])
            dest = path_elastic_fixed / "annotations" / ann_folder / ann
            dest.parent.mkdir(parents = True, exist_ok = True)
            pd.DataFrame(an_tr).to_csv(dest, index = False)
    # affine fixed to moving
    ann_folder = list(fixed_pairing.loc[idx["fixed"]])[1]
    if ann_folder != "":
        for ann in os.listdir(path_elastic_fixed / "annotations" / ann_folder):
            an = pd.read_csv(
                path_elastic_fixed / "annotations" / ann_folder / ann
            )
            an = np.array(an)
            # transform coordinates
            an_ = stack_col_of_ones(an).transpose()


            def theta(theta):
                """
                Get affine transformation matrix from Airlab in the right format.
                """
                if theta.shape == (2, 3):
                    theta = np.vstack((theta, [0, 0, 1]))
                theta = np.linalg.inv(theta)
                return theta[0:2, :]


            # invert A
            A_inv = theta(A)
            an_warped = A_inv @ an_
            an_warped = np.round(
                an_warped.astype(np.int32).transpose(), decimals = 0
            )
            # save warped annotation
            dest = path_affine_fixed / "annotations" / ann_folder / ann
            dest.parent.mkdir(parents = True, exist_ok = True)
            pd.DataFrame(an_warped).to_csv(dest, index = False)

    #################################################
    # plot annotations MOVING -> FIXED ON FIXED IMAGE
    fig, ax = plt.subplots(figsize = (15, 15))
    ax.set_title("moving_{}__fixed_{}".format(f_name[0], f_name[1]))
    ax.imshow(fixed_image, cmap = "gray")

    # plot original
    ann_folder = list(fixed_pairing.loc[idx["fixed"]])[1]
    if ann_folder != "":
        for ann_idx, ann in enumerate(
                os.listdir(fixed_annotations / ann_folder)
        ):
            if (
                    subset_fixed_annotations is not None
                    and ann not in subset_fixed_annotations
            ):
                continue
            an = pd.read_csv(fixed_annotations / ann_folder / ann)
            an_np = np.array(an)
            an = an_np - ofset_fixed
            coord = an.tolist()
            coord.append(
                coord[0]
            )  # repeat the first point to create a 'closed loop'
            xs, ys = zip(*coord)  # create lists of x and y values
            if ann_idx == 0:
                ax.plot(xs, ys, label = "Annotation fixed", color = "green")
            else:
                ax.plot(xs, ys, label = "_nolegend_", color = "green")
    # plot affine
    ann_folder = list(moving_pairing.loc[idx["moving"]])[1]
    if ann_folder != "":
        for ann_idx, ann in enumerate(
                os.listdir(path_affine / "annotations" / ann_folder)
        ):
            if (
                    subset_moving_annotations is not None
                    and ann not in subset_moving_annotations
            ):
                continue
            an = pd.read_csv(path_affine / "annotations" / ann_folder / ann)
            an_np = np.array(an)
            an = an_np
            coord = an.tolist()
            coord.append(
                coord[0]
            )  # repeat the first point to create a 'closed loop'
            xs, ys = zip(*coord)  # create lists of x and y values
            if ann_idx == 0:
                ax.plot(
                    xs, ys, label = "Annotation moving + affine", color = "yellow"
                )
            else:
                ax.plot(xs, ys, label = "_nolegend_", color = "yellow")
    # plot elastic
    ann_folder = list(moving_pairing.loc[idx["moving"]])[1]
    if ann_folder != "":
        for ann_idx, ann in enumerate(
                os.listdir(path_elastic / "annotations" / ann_folder)
        ):
            if (
                    subset_moving_annotations is not None
                    and ann not in subset_moving_annotations
            ):
                continue
            an = pd.read_csv(path_elastic / "annotations" / ann_folder / ann)
            an_np = np.array(an)
            an = an_np
            coord = an.tolist()
            coord.append(
                coord[0]
            )  # repeat the first point to create a 'closed loop'
            xs, ys = zip(*coord)  # create lists of x and y values
            if ann_idx == 0:
                ax.plot(
                    xs,
                    ys,
                    label = "Annotation moving + affine + elastic",
                    color = "orange",
                )
            else:
                ax.plot(xs, ys, label = "_nolegend_", color = "orange")

    # print the plot
    ax.legend()
    dest = (
            path_results
            / "results"
            / "annotations_moving_to_fixed_transfer"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".png")
    )
    dest.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(
        str(dest), bbox_inches = "tight",
    )
    if plot_results:
        fig.show()
    else:
        plt.close(fig)

    ##################################################
    # plot annotations FIXED -> MOVING ON MOVING IMAGE
    fig, ax = plt.subplots(figsize = (15, 15))
    ax.set_title("moving_{}__fixed_{}".format(f_name[0], f_name[1]))
    moving = str(moving_images / f[0])
    moving_image = np.array(Image.open(moving))
    ax.imshow(moving_image, cmap = "gray")
    # plot original
    ann_folder = list(moving_pairing.loc[idx["moving"]])[1]
    if ann_folder != "":
        for ann_idx, ann in enumerate(
                os.listdir(moving_annotations / ann_folder)
        ):
            if (
                    subset_moving_annotations is not None
                    and ann not in subset_moving_annotations
            ):
                continue
            an = pd.read_csv(moving_annotations / ann_folder / ann)
            an_np = np.array(an)
            an = an_np - ofset_moving
            coord = an.tolist()
            coord.append(
                coord[0]
            )  # repeat the first point to create a 'closed loop'
            xs, ys = zip(*coord)  # create lists of x and y values
            if ann_idx == 0:
                ax.plot(xs, ys, label = "Annotation moving", color = "green")
            else:
                ax.plot(xs, ys, label = "_nolegend_", color = "green")
    # # plot elastic
    # ann_folder = list(fixed_pairing.loc[idx['fixed']])[1]
    # if ann_folder != '':
    #     for ann_idx, ann in enumerate(os.listdir(
    #         path_elastic_fixed / "annotations" / ann_folder
    #     )):
    # if subset_moving_annotations is not None and ann not in subset_moving_annotations:
    #   continue
    #         an = pd.read_csv(
    #             path_elastic_fixed / "annotations" / ann_folder / ann
    #         )
    #         an_np = np.array(an)
    #         an = an_np
    #         coord = an.tolist()
    #         coord.append(
    #             coord[0]
    #         )  # repeat the first point to create a 'closed loop'
    #         xs, ys = zip(*coord)  # create lists of x and y values
    #         if ann_idx == 0:
    #             ax.plot(xs, ys, label="Annotation moving - elastic", color="orange")
    #         else:
    #             ax.plot(xs, ys, label="_nolegend_", color="orange")
    # plot affine
    ann_folder = list(fixed_pairing.loc[idx["fixed"]])[1]
    if ann_folder != "":
        for ann_idx, ann in enumerate(
                os.listdir(path_affine_fixed / "annotations" / ann_folder)
        ):
            if (
                    subset_fixed_annotations is not None
                    and ann not in subset_fixed_annotations
            ):
                continue
            an = pd.read_csv(
                path_affine_fixed / "annotations" / ann_folder / ann
            )
            an_np = np.array(an)
            an = an_np
            coord = an.tolist()
            coord.append(
                coord[0]
            )  # repeat the first point to create a 'closed loop'
            xs, ys = zip(*coord)  # create lists of x and y values
            if ann_idx == 0:
                ax.plot(
                    xs,
                    ys,
                    label = "Annotation fixed - elastic - affine",
                    color = "yellow",
                )
            else:
                ax.plot(xs, ys, label = "_nolegend_", color = "yellow")
    # print the plot
    ax.legend()
    dest = (
            path_results
            / "results"
            / "annotations_fixed_to_moving_transfer"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".png")
    )
    dest.parent.mkdir(parents = True, exist_ok = True)
    fig.savefig(
        str(dest), bbox_inches = "tight",
    )
    if plot_results:
        fig.show()
    else:
        plt.close(fig)

####################
# save losses to csv
pd.DataFrame(losses).to_csv(path_results / "losses.csv", index = False)
