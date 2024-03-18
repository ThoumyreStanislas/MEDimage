import logging
import os
import pickle
import sys
import matplotlib.pyplot as plt
from typing import Dict, List, Union
from copy import deepcopy
from pathlib import Path
import nibabel as nib
import argparse

import numpy as np
import pandas as pd

DESCRIPTION = """
    extraction of ngtdm metrics
    """

def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=DESCRIPTION)

    p.add_argument('data',
                   help='niftii image')

    return p

def main():
        
    parser = buildArgsParser()
    args = parser.parse_args()

    data_img = nib.load(args.data)
    affine = data_img.affine
    
    MODULE_DIR = os.path.dirname(os.path.abspath('../MEDimage'))
    sys.path.append(MODULE_DIR)

    import MEDimage

    # À modifier: Chemin vers le fichier de scan STS 009 CT par exemple
    path_read = Path(os.getcwd()) / "data" / "TEP" 

    # Chemin vers le fichier de paramètres JSON
    path_to_params = Path(os.getcwd())  / "settings" / "TEP" / "ngtdm_param_TEP.json"
    im_params = MEDimage.utils.load_json(path_to_params)


    with open(path_read / "STS-IBSI-009__PET.PTscan.npy", 'rb') as f: medscan = pickle.load(f)

    medscan = MEDimage.MEDscan(medscan)

    if medscan.type == 'PTscan':
        medscan.data.volume.array = MEDimage.processing.compute_suv_map(medscan.data.volume.array, medscan.dicomH[0])

    # Init processing & computation parameters
    medscan.init_params(im_params)

    # Nom des régions d'intérêt à trouver dans : medscan.data.ROI.roi_names
    roi_name = "{GTV_Mass_PET}" # àa c'est bon pour STS-IBSI-009 CT

    vol_obj_init, roi_obj_init = MEDimage.processing.get_roi_from_indexes(
    medscan,
    name_roi=roi_name,
    box_string="box10"
    )

    # Interpolation
    # Intensity Mask
    vol_obj = MEDimage.processing.interp_volume(
    medscan=medscan,
    vol_obj_s=vol_obj_init,
    vox_dim=medscan.params.process.scale_non_text,
    interp_met=medscan.params.process.vol_interp,
    round_val=medscan.params.process.gl_round,
    image_type='image',
    roi_obj_s=roi_obj_init,
    box_string=medscan.params.process.box_string
    )
    # Morphological Mask
    roi_obj_morph = MEDimage.processing.interp_volume(
    medscan=medscan,
    vol_obj_s=roi_obj_init,
    vox_dim=medscan.params.process.scale_non_text,
    interp_met=medscan.params.process.roi_interp,
    round_val=medscan.params.process.roi_pv,
    image_type='roi', 
    roi_obj_s=roi_obj_init,
    box_string=medscan.params.process.box_string
    )

    # Re-segmentation
    # Intensity mask range re-segmentation
    roi_obj_int = deepcopy(roi_obj_morph)
    roi_obj_int.data = MEDimage.processing.range_re_seg(
    vol=vol_obj.data, 
    roi=roi_obj_int.data,
    im_range=medscan.params.process.im_range
    )

    # Intensity mask outlier re-segmentation
    roi_obj_int.data = np.logical_and(
    MEDimage.processing.outlier_re_seg(
    vol=vol_obj.data, 
    roi=roi_obj_int.data, 
    outliers=medscan.params.process.outliers
    ),
    roi_obj_int.data
    ).astype(int)

    vol_int_re = MEDimage.processing.roi_extract(
    vol=vol_obj.data, 
    roi=roi_obj_int.data
    )

    image_original = vol_int_re


    # C'est ici que tout se passe!!!
    vol_obj_all_features = MEDimage.filters.apply_filter(
    medscan, 
    vol_int_re, 
    user_set_min_val=medscan.params.process.im_range[0]
    )

    image_coarseness = vol_obj_all_features[:,:,:,0]
    image_contrast = vol_obj_all_features[:,:,:,1]
    image_busyness = vol_obj_all_features[:,:,:,2]
    image_complexity = vol_obj_all_features[:,:,:,3]
    image_strength = vol_obj_all_features[:,:,:,4]

    #features_img = nib.Nifti1Image(image_original.astype(np.float32), affine)
    #nib.save(features_img, "image_original")

    coarseness_img = nib.Nifti1Image(image_coarseness.astype(np.float32), affine)
    nib.save(coarseness_img, "image_TEP_coarseness_g5_FBN_50")

    contrast_img = nib.Nifti1Image(image_contrast.astype(np.float32), affine)
    nib.save(contrast_img, "image_TEP_contrast_g5_FBN_50")

    busyness_img = nib.Nifti1Image(image_busyness.astype(np.float32), affine)
    nib.save(busyness_img, "image_TEP_busyness_g5_FBN_50")

    complexity_img = nib.Nifti1Image(image_complexity.astype(np.float32), affine)
    nib.save(complexity_img, "image_TEP_g5_complexity_FBN_50")

    strength_img = nib.Nifti1Image(image_strength.astype(np.float32), affine)
    nib.save(strength_img, "image_TEP_g5_strength_FBN_50")

if __name__ == "__main__":   
    main()