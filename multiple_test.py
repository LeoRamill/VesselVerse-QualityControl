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

from utils_test import get_val_transform, ensure_portrait, to_uint8_rgb, generate_mips_from_nifti, extract_features_from_nifti

try:
    import VESSEL_METRICS
    from VESSEL_METRICS import process, get_component_rows_from_results, _aggregate, ALL_METRIC_KEYS
except ImportError:
    print("ERRORE CRITICO: 'VESSEL_METRICS.py' non trovato.")
    exit(1)

# Costant ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



def process_single_case(nifti_path, model, device, scaler, tab_cols, transform, args):
    """
    Process a single NIfTI case: generate inputs, run inference, GradCAM, and create output image.
    
    Parameters:
        nifti_path: Path file NIfTI
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
    
    # Image Inputs
    inputs = {}
    mips = None
    
    if args.selected_model in ['multi_CNN', 'multimodal']:
        mips = generate_mips_from_nifti(nifti_path)
        if mips is None: return None, None, None # Error reading NIfTI
        
        for view in ['axial', 'sagittal', 'coronal']:
            img_pil = Image.fromarray(mips[view])
            inputs[view] = transform(img_pil).unsqueeze(0).to(device)

    # Tabular Inputs
    if args.selected_model in ['MLP_tabular', 'multimodal']:
        raw_feats = extract_features_from_nifti(nifti_path, tab_cols)
        if raw_feats is None: return None, None, None # Error feature or Raw file
        
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
        viz_ax = overlay_heatmap_original_size(mips['axial'], cam_ax)
        viz_sag = overlay_heatmap_original_size(mips['sagittal'], cam_sag)
        viz_cor = overlay_heatmap_original_size(mips['coronal'], cam_cor)

        # Resize to same height
        h_max = max(viz_ax.shape[0], viz_sag.shape[0], viz_cor.shape[0])
        
        def resize_h(img, target_h):
            h, w = img.shape[:2]
            if h == target_h: return img
            scale = target_h / float(h)
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

        final_ax = resize_h(viz_ax, h_max)
        final_sag = resize_h(viz_sag, h_max)
        final_cor = resize_h(viz_cor, h_max)

        # Montage
        spacer = np.zeros((h_max, 30, 3), dtype=np.uint8)
        montage = np.hstack([final_ax, spacer, final_sag, spacer, final_cor])
        
        # Add text
        montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
        color = (0, 0, 255) if pred == 1 else (0, 255, 0)
        txt = f"Pred: {pred} | Prob: {prob:.3f}"
        cv2.putText(montage_bgr, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
    return pred, prob, montage_bgr


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
    
    print("Fitting scaler...")
    df_train = pd.read_excel(args.excel_path)
    drop_cols = {"file_sorgente", "label1", "label2"}
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    tab_cols = [c for c in numeric_cols if c not in drop_cols]
    
    scaler = StandardScaler()
    scaler.fit(df_train[tab_cols].astype(np.float32).values)
    tabular_dim = len(tab_cols)
    print(f"Scaler ready. Features: {tabular_dim}")

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
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    model.eval()
    transform = get_val_transform()

    # Gather NIfTI files
    search_path = os.path.join(args.input_folder, "*.nii.gz")
    files = glob.glob(search_path)
    if not files:
        print(f"No .nii.gz files found in {args.input_folder}")
        return

    print(f"Found {len(files)} cases to analyze.")

    # Process each file
    results_list = []
    
    # Progress bar with tqdm if available
    iterator = tqdm(files, desc="Processing") if 'tqdm' in globals() else files

    for nifti_path in iterator:
        filename = os.path.basename(nifti_path)
        
        # Process single case
        pred, prob, img_result = process_single_case(
            nifti_path, model, device, scaler, tab_cols, transform, args
        )
        
        if pred is not None:
            # Save Results
            results_list.append({
                "Filename": filename,
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
            print(f"Skipped {filename} (error during processing).")

    # Save Summary Excel
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
    parser = argparse.ArgumentParser(description="Batch Test Script")
    
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with NIfTI files to process")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder where to save images and excel")
    
 
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path of the model .pth")
    parser.add_argument("--excel_path", type=str, required=True, help="Original Excel for the scaler")
    parser.add_argument("--selected_model", type=str, default="multimodal")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=256)

    args = parser.parse_args()
    
    run_batch_test(args)