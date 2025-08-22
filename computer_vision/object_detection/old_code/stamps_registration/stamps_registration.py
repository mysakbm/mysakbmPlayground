import os
from datetime import date
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
from PIL import Image
import PIL.ImageOps


import src.affine_registration
import src.elastic_registration
import src.master_code_settings as mc_settings
from src.utils import create_square_grid, renormalize, warp_rgb_image

# ENVIROMENT ===================================================================
np.random.seed(0)
th.manual_seed(0)
# noinspection PyUnresolvedReferences
th.backends.cudnn.deterministic = True
# noinspection PyUnresolvedReferences
th.backends.cudnn.benchmark = False


# AUX FUNCTIONS ================================================================


def stack_col_of_ones(arr):
    return np.hstack((arr, np.ones((len(arr), 1))))


def set_registration_parameters(
    config=None,
    results_path=None,
    base_moving_folder=None,
    base_fixed_folder=None,
    moving_images_path=None,
    moving_images_annotations_path=None,
    moving_images_pairing=None,
    fixed_images_path=None,
    fixed_images_annotations_path=None,
    fixed_images_pairing=None,
    channel_moving=0,
    channel_fixed=2,
    image_matching_indices=None,
    affine_registration_method="affine",
    elastic_registration_method="diffeomorphic_bspline",
    gpu_id=0,
    save_loss_csv=False,
):
    """Author: Martin Vagenknecht
    Function sets parameters for registration.

    Parameters:
        config : json-like dictionary
        results_path : str or Path, required
        base_moving_folder : str or Path, required
        base_fixed_folder : str or Path, required
        moving_images_path : str or Path, required
        moving_images_annotations_path : str or Path, required
        moving_images_pairing : str or Path, required
        fixed_images_path : str or Path, required
        fixed_images_annotations_path : str or Path, required
        fixed_images_pairing : str or Path, required
        channel_moving : str or Path, required
        channel_fixed : str or Path, required
        image_matching_indices : str or Path, required
        affine_registration_method : str, optional
            default = \"affine\"
        elastic_registration_method : str, optional
            default = \"diffeomorphic_bspline\"
        gpu_id : str or Path, required
        save_loss_csv : bool, optional
            Flag for saving loss function values
            default = False


    Returns:
      config : json-like dictionary

    """

    if config is None:
        raise Exception("No config provided!")

    # Subset annotations
    if "subset_fixed_annotations" not in config["init_info"]:
        subset_fixed_annotations = None
    elif config["init_info"]["subset_fixed_annotations"] == "None":
        subset_fixed_annotations = None
    else:
        subset_fixed_annotations = config["init_info"][
            "subset_fixed_annotations"
        ]

    if "subset_moving_annotations" not in config["init_info"]:
        subset_moving_annotations = None
    elif config["init_info"]["subset_moving_annotations"] == "None":
        subset_moving_annotations = None
    else:
        subset_moving_annotations = config["init_info"][
            "subset_moving_annotations"
        ]

    # moving folder setting
    if moving_images_path is not None:
        pass
    elif base_moving_folder is not None:
        moving_images_path = str(Path(base_moving_folder) / "images")
    else:
        raise Exception("No moving folder images provided")

    if moving_images_pairing is not None:
        pass
    elif base_moving_folder is not None:
        moving_images_pairing = str(Path(base_moving_folder) / "pairing.csv")
    else:
        raise Exception("No moving folder pairing provided")

    if moving_images_annotations_path is not None:
        pass
    elif base_moving_folder is not None:
        moving_images_annotations_path = str(
            Path(base_moving_folder) / "annotations"
        )
    else:
        raise Exception("No moving folder annotations provided")

    # fixed folder setting
    if fixed_images_path is not None:
        pass
    elif base_fixed_folder is not None:
        fixed_images_path = str(Path(base_fixed_folder) / "images")
    else:
        raise Exception("No fixed folder images provided")

    if fixed_images_pairing is not None:
        pass
    elif base_fixed_folder is not None:
        fixed_images_pairing = str(Path(base_fixed_folder) / "pairing.csv")
    else:
        raise Exception("No fixed folder pairing provided")

    if fixed_images_annotations_path is not None:
        pass
    elif base_fixed_folder is not None:
        fixed_images_annotations_path = str(
            Path(base_fixed_folder) / "annotations_named_complete_csv"
        )
    else:
        raise Exception("No fixed folder annotations provided")

    results_experiment_name = (
        config["init_info"]["study_id"]
        + "_"
        + config["init_info"]["stain"]
        + "_"
        + config["init_info"]["region"]
        + "_"
        + config["init_info"]["slide_number"]
        + "_"
        + config["init_info"]["layer_atlas"]
    )

    if results_path is None:
        results_path = str(
            mc_settings.machine_specific_default_path()
            / "registration_results"
            / str(date.today())
            / results_experiment_name
        )

    if len(str(results_path)) >= 240:
        results_path = str(
            mc_settings.machine_specific_default_path()
            / "registration_results"
            / str(date.today())
            / results_experiment_name
        )

        print(
            "Registration result path is too long. Will be shortened to"
            " default path: " + results_path + ". Later, we will try to "
            "copy the results into experiment folder as well."
        )

    version = 2
    results_path_appr = results_path
    while os.path.exists(results_path_appr):
        results_path_appr = results_path + "_v" + str(version)
        version += 1

    parameters = {
        "results_path": results_path_appr,
        "moving_images_path": moving_images_path,
        "moving_images_annotations_path": moving_images_annotations_path,
        "moving_images_pairing": moving_images_pairing,
        "fixed_images_path": fixed_images_path,
        "fixed_images_annotations_path": fixed_images_annotations_path,
        "fixed_images_pairing": fixed_images_pairing,
        "channel": {"moving": channel_moving, "fixed": channel_fixed},
        "image_matching": image_matching_indices,
        "affine_registration_method": affine_registration_method,
        "elastic_registration_method": elastic_registration_method,
        "gpu_id": gpu_id,
        "save_loss_csv": save_loss_csv,
        "subset_moving_annotations": subset_moving_annotations,
        "subset_fixed_annotations": subset_fixed_annotations,
    }

    return parameters


def test_result_folder_lens(path):
    if len(str(path)) >= 200:
        raise Exception(
            "\n Path: " + path + " is too long. Code will fail. The"
            " MAX_PATH is 256 chars :("
        )

    return "OK"


# MAIN FUNCTION ================================================================
def registration_wrapper(
    results_path,
    moving_images_path,
    moving_images_annotations_path,
    moving_images_pairing,
    fixed_images_path,
    fixed_images_annotations_path,
    fixed_images_pairing,
    channel,
    image_matching,
    affine_registration_method,
    elastic_registration_method,
    gpu_id,
    save_loss_csv,
    subset_moving_annotations,
    subset_fixed_annotations,
):
    """Author: Martin Vagenknecht
    Function for registration of moving image on fixed image. It creates
    folder structure with registrated images. Returns nothing.

    Parameters:
        results_path : str or Path, required,
        moving_images_path : str or Path, required,
        moving_images_annotations_path : str or Path, required,
        moving_images_pairing : str or Path, required,
        fixed_images_path : str or Path, required,
        fixed_images_annotations_path : str or Path, required,
        fixed_images_pairing : str or Path, required,
        channel : dict, required,
            Channel for moving and fixed image.
            default = {"moving": 0, "fixed": 2}
        image_matching : list of dictionaries, required,
            Matching indices of moving and fixed pairing.csv files.
        affine_registration_method : str, required,
            Self explanatory,
            Default = "affine"
        elastic_registration_method : str, required,
            Self explanatory.
            Default = "diffeomorphic_bspline"
        gpu_id: int, required,
            Index which gpu core to use
            default = 0
        save_loss_csv: Bool
            Flag for saving values of loss function
        subset_moving_annotations: list, optional
            a subset of moving annotations
        subset_fixed_annotations: list, optional
            a subset of fixed annotations

    Returns :
        None
    """

    # Test lenghts of paths, there is internal MAX_PATH set to 256 chars in WIN10
    paths_df = pd.Series(
        [
            results_path,
            moving_images_path,
            moving_images_annotations_path,
            moving_images_pairing,
            fixed_images_path,
            fixed_images_annotations_path,
            fixed_images_pairing,
        ]
    )

    paths_df.apply(test_result_folder_lens)

    if th.cuda.is_available():
        device = th.device("cuda:{}".format(str(gpu_id)))
    else:
        device = th.device("cpu")

    # set the used data type
    dtype = th.float32
    # moving
    # annotation mapping table
    moving_pairing = pd.read_csv(
        Path(moving_images_pairing),
        index_col=0,
        na_values=[None],
        keep_default_na=False,
    )

    # fixed
    # annotation mapping table
    fixed_pairing = pd.read_csv(
        Path(fixed_images_pairing),
        index_col=0,
        na_values=[None],
        keep_default_na=False,
    )
    # placeholder for losses
    losses = []
    # RUN LOOP
    for idx in image_matching:
        f = (
            list(moving_pairing.loc[idx["moving"]])[0],
            list(fixed_pairing.loc[idx["fixed"]])[0],
        )

        f_name = (os.path.splitext(f[0])[0], os.path.splitext(f[1])[0])
        folder_name_results = "moving_{}__fixed_{}".format(
            f_name[0], f_name[1]
        )

        # where to save affine registered images
        path_affine = Path(results_path) / folder_name_results / "affine"

        # where to save elastic registered images
        path_elastic = Path(results_path) / folder_name_results / "elastic"

        # where to save fixed to moving affine registered images
        path_affine_fixed = Path(results_path) / folder_name_results / "affine"

        # where to save fixed to moving elastic registered images
        path_elastic_fixed = (
            Path(results_path) / folder_name_results / "elastic"
        )

        # moving_path
        moving = str(Path(moving_images_path) / f[0])

        # fixed_path
        fixed = str(Path(fixed_images_path) / f[1])

        # Check for existence of the images
        if not os.path.exists(fixed):
            print(fixed)
            raise Exception("Fixed file do not exists!")

        if not os.path.exists(moving):
            print(moving)
            raise Exception("Moving file do not exists!")

        # get ofset values for image (coordinates of top left corner)
        ofset_moving = list(
            map(int, os.path.splitext(f[0])[0].split("xx")[1].split("_"))
        )

        ofset_fixed = list(
            map(int, os.path.splitext(f[1])[0].split("xx")[1].split("_"))
        )

        # COPY FILES ===========================================================
        # MOVING SOURCE DATA
        # copy moving image
        dest = (
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "moving"
            / "images"
            / f[0]
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        a = Image.open(moving).convert("RGB")
        a = PIL.ImageOps.mirror(a)
        a.save(dest)


        # FIXED SOURCE DATAFRAME
        target_size = Image.open(moving).size
        original_size = Image.open(fixed).size
        r = [x[1] / x[0] for x in zip(original_size, target_size)]
        # resize and copy fixed image
        dest = (
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "fixed"
            / "images"
            / f[1]
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.open(fixed).resize(target_size).save(dest)
        # resize and copy fixed annotation
        ann_folder = list(fixed_pairing.loc[idx["fixed"]])[1]
        if ann_folder != "":
            for ann in os.listdir(
                Path(fixed_images_annotations_path) / ann_folder
            ):
                if (
                    subset_fixed_annotations is not None
                    and ann.split("_")[0] not in subset_fixed_annotations
                ):
                    continue
                an = pd.read_csv(
                    Path(fixed_images_annotations_path) / ann_folder / ann,
                    header=None
                )
                an_np = np.array(an)
                an = an_np - ofset_fixed
                an = an * r
                an = an.astype(int)
                dest = (
                    Path(results_path)
                    / folder_name_results
                    / "source_data"
                    / "fixed"
                    / "annotations"
                    / ann_folder
                    / ann
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(an).to_csv(dest, index=False, header=False)
        # copy fixed pairing
        mc_settings.copyfile(
            fixed_images_pairing,
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "fixed"
            / "pairing.csv",
        )

        # CHANGE PATHS TO IMAGES AND ANNOTATIONS BASED ON SOURCE DATA
        # moving_path
        moving = str(
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "moving"
            / "images"
            / f[0]
        )

        # fixed_path
        fixed = str(
            Path(results_path)
            / folder_name_results
            / "source_data"
            / "fixed"
            / "images"
            / f[1]
        )


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
        print(
            "================================================================="
        )
        print((f, idx))

        #####################
        # AFFINE REGISTRATION
        # run affine registration
        print("Affine")
        affine_registration = getattr(
            src.affine_registration, affine_registration_method
        )

        displacement, A, loss_affine = affine_registration(
            fixed, moving, channel, device, dtype, plot=False
        )

        # warped
        warped_image = np.array(Image.open(moving))

        warped_image = warp_rgb_image(
            warped_image, displacement, dtype, device
        )

        warped = path_affine / f[0]

        warped.parent.mkdir(parents=True, exist_ok=True)

        Image.fromarray(warped_image).save(warped)

        # moving
        moving_image = np.array(Image.open(moving))

        # fixed
        fixed_image = np.array(Image.open(fixed))

        # make gifs
        # affine vs fixed
        im = Image.fromarray(warped_image)
        im2 = Image.fromarray(fixed_image)
        dest = (
            Path(results_path)
            / folder_name_results
            / "figures"
            / "warped(a)_vs_fixed"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".gif")
        )
        dest.parent.mkdir(parents=True, exist_ok=True)

        im.save(dest, save_all=True, append_images=[im2], duration=700, loop=0)

        # moving vs fixed
        im = Image.fromarray(moving_image)
        im2 = Image.fromarray(fixed_image)
        dest = (
            Path(results_path)
            / folder_name_results
            / "figures"
            / "moving_vs_fixed"
            / str("moving_{}__fixed_{}".format(f_name[0], f_name[1]) + ".gif")
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, save_all=True, append_images=[im2], duration=700, loop=0)


        ####################
        # save losses to csv
        if save_loss_csv:
            pd.DataFrame(losses).to_csv(
                Path(results_path) / "losses.csv", index=False
            )

        print("Registration done")

    return None
