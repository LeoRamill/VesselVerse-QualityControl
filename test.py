import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
import nibabel as nib
from torchvision import transforms

from models.multi_resnet import MultiViewResNet
from models.multi_modal_resnet import MultiModalMultiViewResNet
from models.mlp import MLP_tabular

from utils_xai import MultiViewGradCAM, overlay_heatmap_original_size
import joblib

from utils_test import get_val_transform, ensure_portrait, to_uint8_rgb, generate_mips_from_nifti, extract_features_from_nifti

try:
    import VESSEL_METRICS
    from VESSEL_METRICS import process, get_component_rows_from_results, _aggregate, ALL_METRIC_KEYS
except ImportError:
    print("ERRORE CRITICO: 'VESSEL_METRICS.py' non trovato.")
    exit(1)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def run_single_test(args):
    """
    Process a single NIfTI case: generate inputs, run inference, GradCAM, and create output image.
    
    Parameters:
        args: Parsed command-line arguments
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Used: {device}")

    # Scaler 
    if not os.path.exists(args.excel_path):
        raise FileNotFoundError(f"Excel file not found: {args.excel_path}")

    print("Reading Excel and fitting scaler...")
    df = pd.read_excel(args.excel_path)
    drop_cols = {"file_sorgente", "label1", "label2"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    tab_cols = [c for c in numeric_cols if c not in drop_cols]
    
    scaler = StandardScaler()
    tabular_dim = len(tab_cols)
    print(f"Training {tabular_dim} tabular features.")

    if args.scaler_path and os.path.exists(args.scaler_path):
        print(f"Loading saved scaler from: {args.scaler_path}")
        scaler = joblib.load(args.scaler_path)
    else:
        scaler.fit(df[tab_cols].astype(np.float32).values)

    # Initialize Model
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
    else:
        raise ValueError(f"Model {args.selected_model} not supported")

    # Load Checkpoint
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    model.eval()

    # Input Generation
    print(f"\nAnalysis Patient: {args.nifti_path}")
    print(f"\nAnalysis Segmentation: {args.mask_path}")
    inputs = {}

    mips = None
    if args.selected_model in ['multi_CNN', 'multimodal']:
        # Generate MIPs
        mips = generate_mips_from_nifti(args.nifti_path, args.mask_path)
        # Preprocess MIPs
        transform = get_val_transform()
        for view in ['axial', 'sagittal', 'coronal']:
            img_pil = Image.fromarray(mips[view])
            inputs[view] = transform(img_pil).unsqueeze(0).to(device)

    if args.selected_model in ['MLP_tabular', 'multimodal']:
        raw_feats = extract_features_from_nifti(args.mask_path, tab_cols)
        feats_scaled = scaler.transform(raw_feats.reshape(1, -1))
        inputs['tabular'] = torch.tensor(feats_scaled, dtype=torch.float32).to(device)

    # Inference e GradCAM
    grad_cam = MultiViewGradCAM(model, device)
    
    with torch.enable_grad():
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float:
                v.requires_grad = True
        
        logits = model(inputs)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

        cam_ax, cam_sag, cam_cor = None, None, None
        if args.selected_model in ['multi_CNN', 'multimodal']:
            cam_ax, cam_sag, cam_cor = grad_cam.generate_maps(inputs)

    # Output Results
    print("\n" + "="*40)
    print(f" Prediction: { 'Good' if pred == 1 else 'Bad'}")
    print(f" Probability: {prob:.4f}")
    print("="*40 + "\n")

    # Create Output Image with GradCAM Overlays
    if cam_ax is not None and mips is not None:
        # Overlay heatmap
        viz_ax = overlay_heatmap_original_size(mips['axial'], cam_ax)
        viz_sag = overlay_heatmap_original_size(mips['sagittal'], cam_sag)
        viz_cor = overlay_heatmap_original_size(mips['coronal'], cam_cor)
        # Resize to same height
        h_max = max(viz_ax.shape[0], viz_sag.shape[0], viz_cor.shape[0])
        
        def pad_centered(img, target_h):
            """
            Pads the image vertically to reach target height, centering the original image.
            
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

        # Padding Image
        final_ax = pad_centered(viz_ax, h_max)
        final_sag = pad_centered(viz_sag, h_max)
        final_cor = pad_centered(viz_cor, h_max)

        # Montage
        montage = np.hstack([final_ax, final_sag, final_cor])
        
        # Convert to BGR for OpenCV and add text
        montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
        color = (0, 0, 255) if pred == 1 else (0, 255, 0)
        txt = f"Pred: {pred} | Prob: {prob:.3f}"
        cv2.putText(montage_bgr, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out_name = f"TestResult_{os.path.basename(args.nifti_path)}.jpg"
        cv2.imwrite(out_name, montage_bgr)
        print(f"Image Saved Correctly: {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single NIfTI Test with GradCAM Visualization")
    parser.add_argument("--nifti_path", type=str, required=True, help="Path al file NIfTI di input")
    parser.add_argument("--mask_path", type=str, required=True, help="Path al file segmentation di input")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path al checkpoint del modello")
    parser.add_argument("--excel_path", type=str, required=True, help="Original Excel")
    parser.add_argument("--scaler_path", type=str, default=None)
    parser.add_argument("--selected_model", type=str, default="multimodal")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=256)

    args = parser.parse_args()
    
    run_single_test(args)