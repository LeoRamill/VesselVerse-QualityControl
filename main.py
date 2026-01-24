import os
import argparse
import random
import numpy as np
import pandas as pd
import nibabel as nib # Important for NIfTI processing
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
import wandb
import cv2 
from torchvision import transforms
import torchvision 
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# Custom imports (Assuming these modules exist in your project structure)
from models.multi_resnet import MultiViewResNet
from models.multi_modal_resnet import MultiModalMultiViewResNet
from models.mlp import MLP_tabular
from preprocess.mra_processing import MRAVesselMultiViewDataset
from utils import poly_lr_scheduler, select_optimizer
from utils_xai import MultiViewGradCAM, overlay_heatmap

try:
    import VESSEL_METRICS 
except ImportError:
    print("WARNING: VESSEL_METRICS.py not found. Tabular feature extraction for single NIfTI will fail.")
    VESSEL_METRICS = None

def generate_mips_from_nifti(nifti_path):
    """
    Load a 3D NIfTI file and generate 3 Maximum Intensity Projections (MIPs):
    Axial, Sagittal, and Coronal, maintaining original resolution.
    Returns a dictionary of numpy uint8 arrays (H, W, 3).
    """
    img_obj = nib.load(nifti_path)
    data = img_obj.get_fdata()
    
    # Standard medical orientation: usually (Sagittal, Coronal, Axial) in array dimensions
    # However, it depends on the NIfTI header. We assume standard (X, Y, Z).
    # Axial: Projection onto Z (axis 2)
    # Sagittal: Projection onto X (axis 0)
    # Coronal: Projection onto Y (axis 1)
    
    mip_ax = np.max(data, axis=2)
    mip_sag = np.max(data, axis=0)
    mip_cor = np.max(data, axis=1)

    # Rotations for correct visualization (Empirical, varies by scanner/preprocessing)
    mip_ax = np.rot90(mip_ax)
    mip_sag = np.rot90(mip_sag)
    mip_cor = np.rot90(mip_cor)

    def to_uint8_rgb(mip):
        # Normalize min-max to 0-255
        mip = mip - mip.min()
        if mip.max() > 0:
            mip = (mip / mip.max()) * 255
        mip = mip.astype(np.uint8)
        # Stack to create RGB channels
        return np.stack([mip, mip, mip], axis=-1)

    return {
        "axial": to_uint8_rgb(mip_ax),
        "sagittal": to_uint8_rgb(mip_sag),
        "coronal": to_uint8_rgb(mip_cor)
    }

def extract_features_from_nifti(nifti_path, feature_names_ordered):
    """
    Uses VESSEL_METRICS.process to calculate metrics on the fly.
    Returns an ordered numpy array matching the model's input expectation.
    """
    if VESSEL_METRICS is None:
        raise ImportError("VESSEL_METRICS module is missing. Cannot extract features.")

    # Define which metrics to compute (Should cover those used during training)
    metrics_to_compute = [
        'total_length', 'num_bifurcations', 'bifurcation_density', 'volume',
        'fractal_dimension', 'lacunarity', 'num_loops', 
        'num_abnormal_degree_nodes', 'mean_loop_length', 'max_loop_length',
        'avg_diameter'
    ]
    
    # Call VESSEL_METRICS processing
    # Note: process returns a dict {component_id: {data...}}
    results = VESSEL_METRICS.process(
        nifti_path, 
        selected_metrics=set(metrics_to_compute), 
        save_conn_comp_masks=False, 
        save_seg_masks=False
    )
    
    # Simple Aggregation (Sum for additive metrics, Weighted Mean for others)
    # We initialize with 0.0
    agg_data = {k: 0.0 for k in feature_names_ordered}
    
    total_len_all = 0.0
    
    for cid, data in results.items():
        # Example aggregation logic
        if 'total_length' in data:
            total_len_all += data['total_length']
        
        # Direct sums
        for key in ['total_length', 'num_bifurcations', 'volume', 'num_loops', 'num_abnormal_degree_nodes']:
             if key in agg_data and key in data:
                 agg_data[key] += data[key]
                 
    # Note: For complex features (like Lacunarity), you should implement the exact 
    # aggregation logic used in your training preprocessing script.
    
    # Construct ordered feature vector
    feature_vector = []
    for name in feature_names_ordered:
        val = agg_data.get(name, 0.0)
        feature_vector.append(val)
        
    return np.array(feature_vector, dtype=np.float32)

def predict_single_nifti(args, nifti_path, model, device, scaler, feature_cols, gt_label=None):
    """
    Standalone function to predict on a new single case.
    Generates prediction, probabilities, and a GradCAM visualization montage.
    """
    print(f"\n--- Processing Single Case: {nifti_path} ---")
    
    # 1. Generate Original MIP Images
    mips_orig = generate_mips_from_nifti(nifti_path) # Dict of numpy arrays (H, W, 3)
    
    # 2. Preprocess Images for Model (Resize -> Tensor -> Norm)
    _, val_transform = get_transforms()
    
    inputs_tensor = {}
    for view in ['axial', 'sagittal', 'coronal']:
        # Convert numpy to PIL for transforms
        img_pil = Image.fromarray(mips_orig[view])
        # Apply transforms (Resize 224x224, Norm) and add batch dimension
        img_tensor = val_transform(img_pil).unsqueeze(0).to(device) 
        inputs_tensor[view] = img_tensor

    # 3. Tabular Extraction and Preprocessing
    # Only needed if model is Multimodal or MLP
    if args.selected_model in ['multimodal', 'MLP_tabular']:
        print("Extracting tabular features...")
        raw_feats = extract_features_from_nifti(nifti_path, feature_cols)
        # Scale features using the scaler fitted on training data
        feats_scaled = scaler.transform(raw_feats.reshape(1, -1))
        inputs_tensor['tabular'] = torch.tensor(feats_scaled, dtype=torch.float32).to(device)

    # 4. Prediction
    model.eval()
    
    # Initialize GradCAM
    grad_cam = MultiViewGradCAM(model, device)
    
    cam_ax, cam_sag, cam_cor = None, None, None

    # Enable gradients for GradCAM calculation
    with torch.enable_grad():
        # Requires_grad must be set on input tensors
        for k, v in inputs_tensor.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float:
                v.requires_grad = True

        logits = model(inputs_tensor)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0
        
        # 5. Generate Grad-CAM (224x224 maps)
        # target_category=None uses the predicted class
        if args.selected_model in ['multi_CNN', 'multimodal']:
             cam_ax, cam_sag, cam_cor = grad_cam.generate_maps(inputs_tensor)

    # 6. Visualization
    # Overlay heatmap on original sized images (requires resizing map to original img size)
    # Note: overlay_heatmap usually handles resizing internally if implemented robustly,
    # otherwise we resize the CAM to match mips_orig.
    
    # Helper to resize and overlay
    def apply_overlay(orig_img, cam_map):
        if cam_map is None: return orig_img
        # Resize cam to orig size
        cam_resized = cv2.resize(cam_map, (orig_img.shape[1], orig_img.shape[0]))
        # Use existing utility or cv2 directly
        return overlay_heatmap(orig_img, cam_resized, alpha=0.5) 

    viz_ax = apply_overlay(mips_orig['axial'], cam_ax)
    viz_sag = apply_overlay(mips_orig['sagittal'], cam_sag)
    viz_cor = apply_overlay(mips_orig['coronal'], cam_cor)

    # 7. Create Horizontal Montage (handling different heights with padding)
    h_max = max(viz_ax.shape[0], viz_sag.shape[0], viz_cor.shape[0])
    
    def pad_img(img, target_h):
        if img.shape[0] == target_h: return img
        # Pad with black at the bottom
        pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
        return np.vstack([img, pad])

    viz_ax = pad_img(viz_ax, h_max)
    viz_sag = pad_img(viz_sag, h_max)
    viz_cor = pad_img(viz_cor, h_max)
    
    # White borders for separation
    border = np.ones((h_max, 10, 3), dtype=np.uint8) * 255
    montage = np.hstack([viz_ax, border, viz_sag, border, viz_cor])

    # 8. Add Text Info
    montage_bgr = cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
    label_text = f"Pred: {pred} | Prob: {prob:.4f}"
    if gt_label is not None:
        label_text += f" | GT: {gt_label}"
    
    color = (0, 255, 0) if (gt_label is None or pred == gt_label) else (0, 0, 255)
    cv2.putText(montage_bgr, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Save Result
    output_filename = f"SinglePrediction_{os.path.basename(nifti_path)}.jpg"
    cv2.imwrite(output_filename, montage_bgr)
    print(f"Prediction: {pred} (Prob: {prob:.4f}). Result saved to {output_filename}")
    
    return pred, prob, montage_bgr

def get_transforms():
    """
    Define data augmentations for training and validation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=10),  
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform

def run_test_pipeline(args, checkpoint_path=None, model_instance=None, save_results=True, output_csv="test_predictions.csv"):
    """
    Run the test pipeline on the provided dataset using a trained model.
    """
    print("Testing Pipeline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data and configure scaler (requires fitting on training data first in a real scenario,
    # or loading a saved scaler. Here we approximate by fitting on test data or assuming passed scaler logic).
    # For brevity, reusing the loading logic similar to main or validation.
    
    # NOTE: In a strictly correct pipeline, you should load the scaler saved during training.
    # Here, we assume the dataset class handles scaling if passed, or we refit.
    # To avoid complexity, we assume model_instance is ready and we just load data.
    
    # Implementation of test loading omitted for brevity, assuming standard DataLoader
    # ...
    # Placeholder return
    return 0.0, 0.0

def validation(model, val_loader, device, criterion, selected_model, log_images=True, images_per_epoch=20, step=None):
    """
    Run validation for one epoch.
    """
    model.eval()
    losses, y_true, y_pred, y_prob = [], [], [], []
    wandb_images = []
    
    # Initialize GradCAM if needed
    grad_cam_engine = None
    if log_images and selected_model in ['multi_CNN', 'multimodal']:
        try:
            grad_cam_engine = MultiViewGradCAM(model, device)
        except Exception as e:
            print(f"Warning: Could not initialize GradCAM: {e}") 

    for batch_idx, batch in enumerate(val_loader):
        is_viz_batch = (batch_idx == 0) and log_images
        context_manager = torch.enable_grad() if (is_viz_batch and grad_cam_engine) else torch.no_grad()
        
        with context_manager:
            if selected_model == 'MLP_tabular':
                inputs = {"tabular": batch["tabular"].to(device)}
            elif selected_model == 'multi_CNN':
                inputs = {k: batch[k].to(device) for k in ["axial", "sagittal", "coronal"]}
            elif selected_model == 'multimodal':
                inputs = {k: batch[k].to(device) for k in ["axial", "sagittal", "coronal", "tabular"]}
            
            labels = batch["label"].to(device)
            
            if is_viz_batch and grad_cam_engine:
                 for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.float:
                        v.requires_grad = True

            logits = model(inputs)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            if is_viz_batch and grad_cam_engine:
                # Visualization logic (simplified)
                pass

            losses.append(loss.item())
            y_true.append(labels.detach().cpu())
            y_pred.append(preds.detach().cpu())
            y_prob.append(probs.detach().cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    y_prob = torch.cat(y_prob).numpy()
    
    val_loss = float(np.mean(losses))
    val_acc = accuracy_score(y_true, y_pred)
    try:
        val_auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except:
        val_auc = float('nan')
        fpr, tpr = None, None

    return val_loss, val_acc, val_auc, (fpr, tpr), y_true, y_pred, y_prob

def training(model, train_loader, val_loader, optimizer, lr, device, num_epochs, save_dir="./checkpoints", use_amp=True, images_per_epoch=20, selected_model='multimodal'):
    """
    Main training loop.
    """
    os.makedirs(save_dir, exist_ok=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        poly_lr_scheduler(optimizer, lr, epoch)
        print(f'Starting Training Epoch {epoch}')
        model.train()
        train_losses = []

        for batch in train_loader:
            if selected_model == 'MLP_tabular':
                inputs = {"tabular": batch["tabular"].to(device)}
            elif selected_model == 'multi_CNN':
                inputs = {k: batch[k].to(device) for k in ["axial", "sagittal", "coronal"]}
            elif selected_model == 'multimodal':
                 inputs = {k: batch[k].to(device) for k in ["axial", "sagittal", "coronal", "tabular"]}
            
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        
        train_loss = float(np.mean(train_losses))
        
        # Validation
        val_loss, val_acc, val_auc, (fpr, tpr), y_true, y_pred, y_prob = validation(
            model, val_loader, device, criterion, selected_model, log_images=True, step=epoch
        )

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc, "val_auc": val_auc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"Saved best model (Acc: {best_val_acc:.4f})")

    return best_val_acc

def main():
    parser = argparse.ArgumentParser()
    # Required Arguments
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of dataset")
    parser.add_argument("--excel_path", type=str, required=True, help="Path to excel file with labels/features")
    parser.add_argument("--selected_model", type=str, required=True) # { 'MLP_tabular','multi_CNN', 'multimodal'}
    parser.add_argument("--split_strategy", type=str, required=True) # {'random', 'group_wise'}
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=256)

    # WandB
    parser.add_argument("--wandb_project", type=str, default="multiview-vessel")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # Optional Test Set Arguments
    parser.add_argument("--test_excel_path", type=str, required=False)
    parser.add_argument("--root_dir_test", type=str, required=False)
    
    # NEW ARGUMENT FOR SINGLE NIFTI PREDICTION
    parser.add_argument("--single_nifti_path", type=str, default=None, help="Path to a single .nii.gz for one-shot prediction")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint (optional, loads best_model by default if training)")

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    # 1. Load Data to determine Tabular Columns and Fit Scaler
    # (Essential even for single inference to ensure consistent scaling)
    df = pd.read_excel(args.excel_path)
    
    # Extract patient IDs for splitting
    ids_all = df["file_sorgente"].astype(str).values
    patient_ids_all = np.array([s.rsplit("_", 1)[0] for s in ids_all])
    idx_all = np.arange(len(df))
    
    # Perform GroupWise split to find training set indices
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_rows, va_rows = next(gss.split(idx_all, groups=patient_ids_all))
    
    # Identify numeric columns
    drop_cols = {"file_sorgente", "label1", "label2"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    tab_cols = [c for c in numeric_cols if c not in drop_cols]
    
    # Fit Scaler on Training Data Only
    print("Fitting scaler on training data...")
    scaler = StandardScaler()
    scaler.fit(df.iloc[tr_rows][tab_cols].astype(np.float32).values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Initialize Model
    print(f"Initializing model: {args.selected_model}")
    if args.selected_model == 'multi_CNN':
        model = MultiViewResNet(backbone_name=args.backbone, pretrained=True, hidden_dim=args.hidden_dim).to(device)
    elif args.selected_model == 'MLP_tabular':
        model = MLP_tabular(tabular_dim=len(tab_cols), tab_emb_dim=64, tab_hidden=128, hidden_layer=args.hidden_dim).to(device)
    elif args.selected_model == 'multimodal':
        model = MultiModalMultiViewResNet(
            tabular_dim=len(tab_cols), backbone_name=args.backbone, pretrained=True, 
            tab_emb_dim=64, tab_hidden=128, fusion_hidden=args.hidden_dim
        ).to(device)
    else:
        raise ValueError("Unknown model type")

    # 3. BRANCH: Single NIfTI Prediction
    if args.single_nifti_path is not None:
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            print(f"Loading checkpoint from {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("WARNING: No checkpoint provided for single inference. Using random/pretrained weights (or weights after training if you implemented that flow).")

        if os.path.exists(args.single_nifti_path):
            predict_single_nifti(
                args=args,
                nifti_path=args.single_nifti_path,
                model=model,
                device=device,
                scaler=scaler,
                feature_cols=tab_cols,
                gt_label=None # You can pass ground truth if known
            )
        else:
            print(f"File not found: {args.single_nifti_path}")
            
        # Exit after single prediction
        return 

    # 4. Standard Training Pipeline (if not single inference)
    train_transform, val_transform = get_transforms()
    
    # Create Datasets
    train_dataset_full = MRAVesselMultiViewDataset(
        root_dir=args.root_dir, excel_path=args.excel_path, label_col=args.label_col,
        tabular_cols=tab_cols, tabular_scaler=scaler, drop_cols=list(drop_cols), transform=train_transform
    )
    val_dataset_full = MRAVesselMultiViewDataset(
        root_dir=args.root_dir, excel_path=args.excel_path, label_col=args.label_col,
        tabular_cols=tab_cols, tabular_scaler=scaler, drop_cols=list(drop_cols), transform=val_transform
    )

    # Subsetting based on pre-calculated indices (tr_rows, va_rows)
    train_ds = Subset(train_dataset_full, tr_rows)
    val_ds = Subset(val_dataset_full, va_rows)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = select_optimizer(args, model)

    # Run Training
    best_val_acc = training(
        model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
        lr=args.lr, device=device, num_epochs=args.epochs, selected_model=args.selected_model,
        save_dir="./checkpoints_multiview"
    )

    print(f"Training Complete. Best Validation Accuracy: {best_val_acc}")
    wandb.finish()

if __name__ == "__main__":
    main()