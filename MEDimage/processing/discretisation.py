#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
from typing import Tuple

import numpy as np

from ..processing.equalization import equalization


def discretisation(vol_re: np.ndarray,
                   discr_type: str,
                   n_q=None,
                   user_set_min_val=None,
                   ivh=False) -> Tuple[np.ndarray, float]:
    """Quantisizes the image intensities inside the ROI.

    Note:
        For 'FBS' type, it is assumed that re-segmentation with
        proper range was already performed

    Args:
        vol_re (ndarray): 3D array of the image volume that will be studied with
                          NaN value for the excluded voxels (voxels outside the ROI mask).
        discr_type (str): Discretisaion approach/type must be: "FBS", "FBN", "FBSequal"
                          or "FBNequal".
        n_q (float): Number of bins for FBS algorithm and bin width for FBN algorithm.
        user_set_min_val (float): Minimum of range re-segmentation for FBS discretisation,
                                  for FBN discretisation, this value has no importance as an argument
                                  and will not be used.
        ivh (bool): Must be set to True for IVH (Intensity-Volume histogram) features.

    Returns:
        2-element tuple containing

        - ndarray: Same input image volume but with discretised intensities.
        - float: bin width.
    """

    # AZ: NOTE: the "type" variable that appeared in the MATLAB source code
    # matches the name of a standard python function. I have therefore renamed
    # this variable "discr_type"

    # PARSING ARGUMENTS
    vol_quant_re = deepcopy(vol_re)

    if n_q is None:
        return None

    if not isinstance(n_q, float):
        n_q = float(n_q)

    if discr_type not in ["FBS", "FBN", "FBSequal", "FBNequal"]:
        raise ValueError(
            "discr_type must either be \"FBS\", \"FBN\", \"FBSequal\" or \"FBNequal\".")

    # DISCRETISATION
    if discr_type in ["FBS", "FBSequal"]:
        if user_set_min_val is not None:
            min_val = deepcopy(user_set_min_val)
        else:
            min_val = np.nanmin(vol_quant_re)
    else:
        min_val = np.nanmin(vol_quant_re)

    max_val = np.nanmax(vol_quant_re)

    if discr_type == "FBS":
        w_b = n_q
        w_d = w_b
        vol_quant_re = np.floor((vol_quant_re - min_val) / w_b) + 1.0
    elif discr_type == "FBN":
        w_b = (max_val - min_val) / n_q
        w_d = 1.0
        vol_quant_re = np.floor(
            n_q * ((vol_quant_re - min_val)/(max_val - min_val))) + 1.0
        vol_quant_re[vol_quant_re == np.nanmax(vol_quant_re)] = n_q
    elif discr_type == "FBSequal":
        w_b = n_q
        w_d = w_b
        vol_quant_re = equalization(vol_quant_re)
        vol_quant_re = np.floor((vol_quant_re - min_val) / w_b) + 1.0
    elif discr_type == "FBNequal":
        w_b = (max_val - min_val) / n_q
        w_d = 1.0
        vol_quant_re = vol_quant_re.astype(np.float32)
        vol_quant_re = equalization(vol_quant_re)
        vol_quant_re = np.floor(
            n_q * ((vol_quant_re - min_val)/(max_val - min_val))) + 1.0
        vol_quant_re[vol_quant_re == np.nanmax(vol_quant_re)] = n_q
    if ivh and discr_type in ["FBS", "FBSequal"]:
        vol_quant_re = min_val + (vol_quant_re - 0.5) * w_b

    return vol_quant_re, w_d
