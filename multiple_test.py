from ast import Return
import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import joblib
from PIL import Image
from sklearn.preprocessing import StandardScaler
import nibabel as nib
from torchvision import transforms
from tqdm import tqdm 

import re
from pathlib import Path

from models.multi_resnet import MultiViewResNet
from models.multi_modal_resnet import MultiModalMultiViewResNet
from models.mlp import MLP_tabular
from utils_xai import MultiViewGradCAM, overlay_heatmap_original_size

from utils_test import get_val_transform, ensure_portrait, to_uint8_rgb, generate_mips_from_nifti, extract_features_from_nifti

# Check for optional dependency
try:
    import VESSEL_METRICS
    from VESSEL_METRICS import process, get_component_rows_from_results, _aggregate, ALL_METRIC_KEYS
except ImportError:
    print("CRITICAL ERROR: 'VESSEL_METRICS.py' not found.")
    exit(1)

# Constant ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def pad_centered(img, target_h):
    """
    Pads the image vertically to reach target height, centering the original image.
    (Logic taken from single_test to ensure consistent visualization)
    
    Parameters:
        img: Input image as a numpy array (H x W x C)
        target_h: Target height after padding
    Returns:
        padded_img: Vertically padded image (target_h x W x C)
    """
    h, w, c = img.shape
    diff = target_h - h
    if diff <= 0:
        return img
    
    top = diff // 2
    bottom = diff - top
    
    # Padding: ((top, bottom), (left, right), (channels))
    return np.pad(img, ((top, bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)

def process_single_case(nifti_path, mask_path, model, device, scaler, tab_cols, transform, args):
    """
    Process a single NIfTI case: generate inputs, run inference, GradCAM, and create output image.
    Reflects the mechanics of 'run_single_test'.
    
    Parameters:
        nifti_path: Path to NIfTI image
        mask_path: Path to NIfTI mask
        model: PyTorch model
        device: Torch device
        scaler: StandardScaler for tabular data
        tab_cols: List of tabular feature names
        transform: torchvision transform for image preprocessing
        args: Parsed command-line arguments
        
    Returns:
        pred: Predicted class (0 or 1)
        prob: Predicted probability (float)
        montage_bgr: Output image with GradCAM overlays (H, W, 3) uint8
    """
    
    inputs = {}
    mips = None
    
    # Image Inputs (MIPs)
    if args.selected_model in ['multi_CNN', 'multimodal']:
        # Generate MIPs using both NIfTI and Mask (as per single file logic)
        mips = generate_mips_from_nifti(nifti_path, mask_path)
        
        if mips is None: 
            return None, None, None # Error reading NIfTI or Mask
        
        for view in ['axial', 'sagittal', 'coronal']:
            img_pil = Image.fromarray(mips[view])
            inputs[view] = transform(img_pil).unsqueeze(0).to(device)

    # Tabular Inputs
    if args.selected_model in ['MLP_tabular', 'multimodal']:
        # Extract features from the MASK (as per single file logic)
        raw_feats = extract_features_from_nifti(mask_path, tab_cols)
        
        if raw_feats is None: 
            return None, None, None # Error extracting features
        
        feats_scaled = scaler.transform(raw_feats.reshape(1, -1))
        inputs['tabular'] = torch.tensor(feats_scaled, dtype=torch.float32).to(device)

    # Inference + GradCAM
    grad_cam = MultiViewGradCAM(model, device)
    cam_ax, cam_sag, cam_cor = None, None, None
    
    with torch.enable_grad():
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float:
                v.requires_grad = True
        
        logits = model(inputs)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

        if args.selected_model in ['multi_CNN', 'multimodal']:
            cam_ax, cam_sag, cam_cor = grad_cam.generate_maps(inputs)

    # Create Output Image with GradCAM Overlays
    montage_bgr = None
    if cam_ax is not None and mips is not None:
        # Overlay heatmap
        viz_ax = overlay_heatmap_original_size(mips['axial'], cam_ax)
        viz_sag = overlay_heatmap_original_size(mips['sagittal'], cam_sag)
        viz_cor = overlay_heatmap_original_size(mips['coronal'], cam_cor)

        # Pad to same height (using single_test logic)
        h_max = max(viz_ax.shape[0], viz_sag.shape[0], viz_cor.shape[0])
        
        final_ax = pad_centered(viz_ax, h_max)
        final_sag = pad_centered(viz_sag, h_max)
        final_cor = pad_centered(viz_cor, h_max)

        # Montage
        montage = np.hstack([final_ax, final_sag, final_cor])
        
        # Add text
        montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
        color = (0, 0, 255) if pred == 1 else (0, 255, 0)
        txt = f"Pred: {pred} | Prob: {prob:.3f}"
        cv2.putText(montage_bgr, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
    return pred, prob, montage_bgr

def path_by_id(folder, file_id):
    """
    Finds the absolute path of a .nii.gz file containing a specific ID.

    Args:
        folder (str): The directory to search in.
        file_id (str or int): The unique ID (e.g., "IXI123").

    Returns:
        str: The complete absolute path to the file.

    Raises:
        FileNotFoundError: If no file matching the ID is found.
    """
    # Create a Path object and search for the pattern
    search_pattern = f"{file_id}.nii.gz"
    matches = list(Path(folder).glob(search_pattern))

    if not matches:
        return None  # or raise FileNotFoundError(f"No file found for ID: {file_id}")

    # Return the absolute path of the first match as a string
    return str(matches[0].resolve())

def run_batch_test(args):
    """
    Run batch testing on a folder of NIfTI files.
    Parameters:
        args: Parsed command-line arguments
        
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Use device: {device}")
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"Output folder: {args.output_folder}")

    # Setup Scaler
    if not os.path.exists(args.excel_path):
        raise FileNotFoundError(f"File Excel training not found: {args.excel_path}")
    
    df_train = pd.read_excel(args.excel_path)
    drop_cols = {"file_sorgente", "label1", "label2"}
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    tab_cols = [c for c in numeric_cols if c not in drop_cols]
    tabular_dim = len(tab_cols)
    print(f"Tabular Features: {tabular_dim}")

    scaler = StandardScaler()
    if args.scaler_path and os.path.exists(args.scaler_path):
        print(f"Loading saved scaler from: {args.scaler_path}")
        scaler = joblib.load(args.scaler_path)
    else:
        print("Fitting scaler from Excel...")
        scaler.fit(df_train[tab_cols].astype(np.float32).values)

    # Load Model
    print(f"Loading model: {args.selected_model}")
    if args.selected_model == 'multi_CNN':
        model = MultiViewResNet(backbone_name=args.backbone, pretrained=False, hidden_dim=args.hidden_dim).to(device)
    elif args.selected_model == 'MLP_tabular':
        model = MLP_tabular(tabular_dim=tabular_dim, tab_emb_dim=64, tab_hidden=128, hidden_layer=args.hidden_dim).to(device)
    elif args.selected_model == 'multimodal':
        model = MultiModalMultiViewResNet(
            tabular_dim=tabular_dim, backbone_name=args.backbone, pretrained=False,
            tab_emb_dim=64, tab_hidden=128, fusion_hidden=args.hidden_dim
        ).to(device)
    
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("Checkpoint loaded.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    model.eval()
    transform = get_val_transform()
    
    # Extract unique IDs from the input folder using Regex (looking for pattern IXI + 3 digits)
    regex_pattern = r'(IXI\d{3}|Normal\d{3}-MRA)'
    
    raw_ids = []
    if os.path.exists(args.input_folder):
        raw_ids = [
            re.search(regex_pattern, fname).group(0) 
            for fname in os.listdir(args.input_folder) 
            if re.search(regex_pattern, fname)
        ]
    
    # Remove duplicates and sort
    raw_ids = sorted(list(set(raw_ids)))

    if not raw_ids:
        print(f"No valid IDs (IXI...) found in {args.input_folder}")
        return

    print(f"Found {len(raw_ids)} unique IDs to analyze.")

    # Process each case based on ID
    results_list = []
    
    # Progress bar with tqdm if available
    iterator = tqdm(raw_ids, desc="Processing") if 'tqdm' in globals() else raw_ids
    
    for case_id in iterator:
        # Find absolute path for the input image using the ID
        nifti_path = path_by_id(args.input_folder, case_id)
        
        if nifti_path is None:
            print(f"Warning: Image file not found for ID {case_id}. Skipping.")
            continue
            
        filename = os.path.basename(nifti_path)

        # Determine mask path using the ID
        # First check the main mask folder
        mask_path = path_by_id(args.mask_folder, case_id)
        
        # If not found and a segmentation model suffix is provided, check the alternative folder
        if (mask_path is None) and (args.segmentation_model is not None) and (len(args.segmentation_model) > 0):
            alt_mask_folder = args.mask_folder + '_' + args.segmentation_model
            if os.path.exists(alt_mask_folder):
                mask_path = path_by_id(alt_mask_folder, case_id)

        if mask_path is None:
            print(f"Warning: Mask not found for ID {case_id}. Skipping.")
            continue

        # Process single case
        pred, prob, img_result = process_single_case(
            nifti_path, mask_path, model, device, scaler, tab_cols, transform, args
        )
        
        if pred is not None:
            # Save Results Data
            results_list.append({
                "Filename": filename,
                "Case_ID": case_id,
                "Prediction": int(pred),
                "Probability": float(prob),
                "Label": "Good" if pred == 1 else "Bad"
            })
            # Save Image
            if img_result is not None:
                out_img_name = filename.replace(".nii.gz", "_result.jpg")
                out_img_path = os.path.join(args.output_folder, out_img_name)
                cv2.imwrite(out_img_path, img_result)
        else:
            # Logging handled inside process_single_case if None is returned
            pass
    # Save Excel Summary
    if results_list:
        df_results = pd.DataFrame(results_list)
        # Sort by Filename
        df_results.sort_values(by="Filename", inplace=True)
        out_excel_path = os.path.join(args.output_folder, "Results_Summary.xlsx")
        df_results.to_excel(out_excel_path, index=False)
        
        print("\n" + "="*50)
        print(f"COMPLETED.")
        print(f"Excel saved in: {out_excel_path}")
        print(f"Images saved in: {args.output_folder}")
        print("="*50)
    else:
        print("No valid results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Test Script with Masks and GradCAM")
    
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with raw NIfTI files")
    parser.add_argument("--mask_folder", type=str, required=True, help="Folder with segmentation NIfTI files")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder where to save images and excel")
    parser.add_argument("--segmentation_model", type=str, required=False, help="Name of the segmentation model used")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path of the model .pth")
    parser.add_argument("--excel_path", type=str, required=True, help="Original Excel for the scaler (features reference)")
    parser.add_argument("--scaler_path", type=str, default=None, help="Path to saved joblib scaler (optional)")
    
    parser.add_argument("--selected_model", type=str, default="multimodal")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=256)

    args = parser.parse_args()
    
    run_batch_test(args)