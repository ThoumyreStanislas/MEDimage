import logging
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = os.path.dirname(os.path.abspath('../MEDimage/MEDimage/'))
sys.path.append(MODULE_DIR)

import MEDimage

# À modifier: Chemin vers le fichier de scan STS 009 CT par exemple
path_read = Path(os.getcwd()) / "MastersProject" / "RccCHUS" / "data" / "rccchus-npy"

# Chemin vers le fichier de paramètres JSON
path_to_params = Path(os.getcwd())  / "MastersProject" / "RccCHUS" / "settings" / "extraction_settings.json"
im_params = MEDimage.utils.load_json(path_to_params)


with open(path_read / "STS-IBSI-009__CT.CTscan.npy", 'rb') as f: medscan = pickle.load(f)

medscan = MEDimage.MEDscan(medscan)

# Init processing & computation parameters
medscan.init_params(im_params)

# Nom des régions d'intérêt à trouver dans : medscan.data.ROI.roi_names
roi_name = "{GTV_Mass_CT}" # àa c'est bon pour STS-IBSI-009 CT

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

# C'est ici que tout se passe!!!
vol_obj_all_features = MEDimage.filters.apply_filter(
    medscan, 
    vol_int_re, 
    user_set_min_val=medscan.params.process.im_range[0]
)