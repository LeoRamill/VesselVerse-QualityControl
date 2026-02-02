import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
import nibabel as nib
from torchvision import transforms
from tqdm import tqdm 
from models.multi_resnet import MultiViewResNet
from models.multi_modal_resnet import MultiModalMultiViewResNet
from models.mlp import MLP_tabular
from utils_xai import MultiViewGradCAM, overlay_heatmap_original_size
from visualization.max_intensity_proj import maximum_intensity_projection, normalize_to_8bit

try:
    import VESSEL_METRICS
    from VESSEL_METRICS import process, get_component_rows_from_results, _aggregate, ALL_METRIC_KEYS
except ImportError:
    print("ERRORE CRITICO: 'VESSEL_METRICS.py' non trovato.")
    exit(1)

# Costant ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_val_transform():
    """
    Transform Validation: Resize, ToTensor, Normalize.
    
    Returns:
        torchvision.transforms.Compose: Composed transform.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def ensure_portrait(img_array):
    """
    Ensure the image array is in portrait orientation (height >= width).
    If width > height, rotate the image by 90 degrees.
    Parameters:
        img_array: Numpy array image (H, W) o (H, W, C)
    Returns:
        img_array: Numpy array image in portrait orientation (H, W) o (H, W, C)
    """
    h, w = img_array.shape[:2]
    if w > h:
        return np.rot90(img_array)
    return img_array

def to_uint8_rgb(mip):
    """
    Convert a MIP image to uint8 RGB format.
    
    Parameters:
        mip: Numpy array MIP image (H, W) float32 o float64
    Returns:
        Numpy array MIP image (H, W, 3) uint8
    """
    mip = mip - mip.min()
    if mip.max() > 0:
        mip = (mip / mip.max()) * 255
    mip = mip.astype(np.uint8)
    return np.stack([mip, mip, mip], axis=-1)

def generate_mips_from_nifti(nifti_path, mask_path):
    """
    Generate MIPs (Maximum Intensity Projections) from a NIfTI file.
    
    Parameters:
        nifti_path: Path al file NIfTI
        mask_path: Path al file di segmentazione NIfTI
    Returns:
        dict: dictionary with keys 'axial', 'sagittal', 'coronal' and values as RGB uint8 images.
    """
    # 
    projections = {
        "axial": "axial",
        "sagittal": "sagittal",
        "coronal": "coronal"
    }
    
    output_dict = {}

    for key, proj_dim in projections.items():
        mip_raw, _ = maximum_intensity_projection(nifti_path, proj_dim, mask_filepath=mask_path)
        
        if mip_raw is None:
            print(f"Failed to generate {key} projection for {os.path.basename(nifti_path)}")
            return None
        if key == "axial":
            mip_raw = mip_raw 
            #mip_raw = np.rot90(mip_raw, k=2)
        else:
            mip_raw = ensure_portrait(mip_raw)
        # Normalize to 8-bit
        mip_uint8 = normalize_to_8bit(mip_raw)

        # Convert to RGB
        mip_rgb = np.stack([mip_uint8, mip_uint8, mip_uint8], axis=-1)

        output_dict[key] = mip_rgb

    return output_dict

def extract_features_from_nifti(nifti_path, feature_names_ordered):
    """
    Extract tabular features from a NIfTI file using VESSEL_METRICS.
    
    Parameters:
        nifti_path: Path file NIfTI
        feature_names_ordered: list of feature names in order
    Returns:
        np.ndarray: Array di features float32
    """
    try:
        # Check anti-raw
        img_tmp = nib.load(nifti_path)
        if len(np.unique(img_tmp.get_fdata())) > 50:
            print(f"SKIP FEATURE: {os.path.basename(nifti_path)} seems to be a raw image.")
            return None

        results = process(
            nifti_path, 
            selected_metrics=set(ALL_METRIC_KEYS), 
            save_conn_comp_masks=False, 
            save_seg_masks=False
        )
        
        if not results:
            return np.zeros(len(feature_names_ordered), dtype=np.float32)

        rows = get_component_rows_from_results(results)
        df_components = pd.DataFrame(rows)
        agg_metrics = _aggregate(df_components)

        feature_vector = []
        for name in feature_names_ordered:
            val = agg_metrics.get(name, 0.0)
            if pd.isna(val): val = 0.0
            feature_vector.append(val)
        
        return np.array(feature_vector, dtype=np.float32)
    except Exception as e:
        print(f"Error extracting features {os.path.basename(nifti_path)}: {e}")
        return None

