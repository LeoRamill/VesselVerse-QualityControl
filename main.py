import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
import wandb
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision 
import cv2 
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, random_split, Dataset, Subset
# --- MODIFICA: Aggiunti f1_score e precision_score ---
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, precision_score

# Import dei modelli e utils originali
from models.multi_resnet import MultiViewResNet
from models.multi_modal_resnet import MultiModalMultiViewResNet
from models.mlp import MLP_tabular
from preprocess.mra_processing import MRAVesselMultiViewDataset
from utils import denorm_to_uint8, make_triplet_figure, log_val_images_to_wandb, poly_lr_scheduler, select_optimizer, select_splitting_strategy

# --- IMPORT PER XAI ---
from utils_xai import MultiViewGradCAM, overlay_heatmap

# Costanti ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    return train_transform, val_transform

def denormalize_tensor(tensor):
    """
    Inverte la normalizzazione ImageNet ma mantiene il tipo TENSOR.
    """
    t = tensor.clone().detach()
    for t_c, m, s in zip(t, IMAGENET_MEAN, IMAGENET_STD):
        t_c.mul_(s).add_(m)
    t = torch.clamp(t, 0, 1)
    return t

def run_test_pipeline(
    args, 
    checkpoint_path=None, 
    model_instance=None, 
    save_results=True,
    output_csv="test_predictions.csv",
    log_images_to_wandb=True
):
    print("Testing Pipeline")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Preparazione Dati Test
    df = pd.read_excel(args.test_excel_path)
    drop_cols = {"file_sorgente", "label1", "label2"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    tab_cols = [c for c in numeric_cols if c not in drop_cols]
    
    ids_all = df["file_sorgente"].astype(str).values
    patient_ids_all = np.array([s.rsplit("_", 1)[0] for s in ids_all])
    idx_all = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
    tr_rows, _ = next(gss.split(idx_all, groups=patient_ids_all))
    
    print("Fitting scaler on available tabular data...")
    scaler = StandardScaler()
    scaler.fit(df.iloc[tr_rows][tab_cols].astype(np.float32).values)
    
    _, val_transform = get_transforms()
    
    test_dataset = MRAVesselMultiViewDataset(
        root_dir=args.root_dir_test,
        excel_path=args.test_excel_path, 
        label_col=args.label_col,
        tabular_cols=tab_cols,
        tabular_scaler=scaler,      
        drop_cols=list(drop_cols),
        transform=val_transform    
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Test dataset samples: {len(test_dataset)}")
    
    # 2. Caricamento Modello
    if model_instance is not None:
        model = model_instance
    elif checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        if args.selected_model == 'multi_CNN':
            model = MultiViewResNet(backbone_name=args.backbone, pretrained=False, hidden_dim=args.hidden_dim)
        elif args.selected_model == 'MLP_tabular':
            model = MLP_tabular(tabular_dim=test_dataset.tabular_dim, tab_emb_dim=64, tab_hidden=128, hidden_layer=args.hidden_dim)
        elif args.selected_model == 'multimodal':
            model = MultiModalMultiViewResNet(
                tabular_dim=test_dataset.tabular_dim, backbone_name=args.backbone, 
                pretrained=False, tab_emb_dim=64, tab_hidden=128, fusion_hidden=args.hidden_dim
            )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("Either checkpoint_path or model_instance must be provided.")

    model.to(device)
    model.eval()
    
    # 3. Setup WandB Table e GradCAM
    wandb_table = None
    grad_cam = None
    
    if log_images_to_wandb and args.selected_model in ['multi_CNN', 'multimodal']:
        print("Initializing WandB Table and GradCAM for full test set logging...")
        wandb_table = wandb.Table(columns=["Patient ID", "Grad-CAM Visualization", "Ground Truth", "Prediction", "Probability"])
        try:
            grad_cam = MultiViewGradCAM(model, device)
        except Exception as e:
            print(f"Could not initialize GradCAM: {e}")
            log_images_to_wandb = False

    all_preds, all_probs, all_labels, all_ids = [], [], [], []

    print(f"Starting inference... Total batches: {len(test_loader)}")
    
    for batch_idx, batch in enumerate(test_loader):
        context_manager = torch.enable_grad() if (log_images_to_wandb and grad_cam) else torch.no_grad()
        
        with context_manager:
            if args.selected_model == 'MLP_tabular':
                inputs = {"tabular": batch["tabular"].to(device)}
            elif args.selected_model == 'multi_CNN':
                inputs = {
                    "axial": batch["axial"].to(device),
                    "sagittal": batch["sagittal"].to(device),
                    "coronal": batch["coronal"].to(device),
                }
            elif args.selected_model == 'multimodal':
                inputs = {
                    "axial": batch["axial"].to(device),
                    "sagittal": batch["sagittal"].to(device),
                    "coronal": batch["coronal"].to(device),
                    "tabular": batch["tabular"].to(device),
                }
            
            if log_images_to_wandb and grad_cam:
                 for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.float:
                        v.requires_grad = True

            labels = batch["label"].to(device)
            ids = batch["id"]

            logits = model(inputs)
            probs = torch.sigmoid(logits).squeeze()
            if probs.ndim == 0: probs = probs.unsqueeze(0)
            preds = (probs >= 0.5).float()

            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            all_ids.extend(ids)

            if log_images_to_wandb and grad_cam:
                batch_size_curr = len(ids)
                for i in range(batch_size_curr):
                    try:
                        single_input = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
                        cam_ax, cam_sag, cam_cor = grad_cam.generate_maps(single_input)
                        
                        img_ax_vis = denormalize_tensor(inputs['axial'][i])
                        img_sag_vis = denormalize_tensor(inputs['sagittal'][i])
                        img_cor_vis = denormalize_tensor(inputs['coronal'][i])

                        viz_ax = overlay_heatmap(img_ax_vis, cam_ax)
                        viz_sag = overlay_heatmap(img_sag_vis, cam_sag)
                        viz_cor = overlay_heatmap(img_cor_vis, cam_cor)
                        
                        border = np.ones((viz_ax.shape[0], 5, 3), dtype=np.uint8) * 255
                        montage = np.hstack([viz_ax, border, viz_sag, border, viz_cor])
                        
                        wandb_table.add_data(
                            str(ids[i]),
                            wandb.Image(montage),
                            int(labels[i].item()),
                            int(preds[i].item()),
                            probs[i].item()
                        )
                    except Exception as e:
                        print(f"Error logging image for {ids[i]}: {e}")

    # Calcolo metriche test
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')
    
    print(f"Test Accuracy:  {acc:.4f}")
    print(f"Test F1 Score:  {f1:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test AUC:       {auc:.4f}")

    if save_results:
        results_df = pd.DataFrame({
            "patient_id": all_ids,
            "true_label": all_labels,
            "predicted_label": all_preds,
            "probability": all_probs
        })
        results_df.to_csv(output_csv, index=False)
        
    if wandb_table is not None:
        print("Uploading Test Set Table to WandB...")
        wandb.log({"Test Set Analysis": wandb_table})

    return acc, auc

def validation(model, val_loader, device, criterion, selected_model, log_images=True, images_per_epoch=20, step=None):
    """
    Validation loop: Calcola Loss, Acc, AUC, F1, Precision e logga immagini GradCAM.
    """
    print('Starting Validation')
    model.eval()
    losses, y_true, y_pred, y_prob = [], [], [], []
    wandb_images = []

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
                inputs = {"axial": batch["axial"].to(device), "sagittal": batch["sagittal"].to(device), "coronal": batch["coronal"].to(device)}
            elif selected_model == 'multimodal':
                inputs = {"axial": batch["axial"].to(device), "sagittal": batch["sagittal"].to(device), "coronal": batch["coronal"].to(device), "tabular": batch["tabular"].to(device)}
            
            labels = batch["label"].to(device)

            if is_viz_batch and grad_cam_engine:
                 for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.float: v.requires_grad = True

            logits = model(inputs)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            if is_viz_batch and grad_cam_engine:
                num_imgs = min(len(labels), images_per_epoch)
                try:
                    for i in range(num_imgs):
                        single_input = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
                        cam_ax, cam_sag, cam_cor = grad_cam_engine.generate_maps(single_input)
                        
                        img_ax_vis = denormalize_tensor(inputs['axial'][i])
                        img_sag_vis = denormalize_tensor(inputs['sagittal'][i])
                        img_cor_vis = denormalize_tensor(inputs['coronal'][i])

                        viz_ax = overlay_heatmap(img_ax_vis, cam_ax)
                        viz_sag = overlay_heatmap(img_sag_vis, cam_sag)
                        viz_cor = overlay_heatmap(img_cor_vis, cam_cor)
                        
                        montage = np.hstack([viz_ax, viz_sag, viz_cor])
                        caption = f"ID: {batch['id'][i]} | GT: {int(labels[i].item())} | Pred: {int(preds[i].item())}"
                        wandb_images.append(wandb.Image(montage, caption=caption))
                except Exception as e:
                    print(f"Error val GradCAM: {e}")

            losses.append(loss.item())
            y_true.append(labels.detach().cpu())
            y_pred.append(preds.detach().cpu())
            y_prob.append(probs.detach().cpu())

    y_true = torch.cat(y_true).numpy().astype(int)
    y_pred = torch.cat(y_pred).numpy().astype(int)
    y_prob = torch.cat(y_prob).numpy()
    
    val_loss = float(np.mean(losses)) if len(losses) else 0.0
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred, average='binary')
    val_precision = precision_score(y_true, y_pred, average='binary', zero_division=0)

    if len(np.unique(y_true)) == 2:
        try:
            val_auc = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, _ = roc_curve(y_true, y_prob)
        except ValueError:
            val_auc, fpr, tpr = float("nan"), None, None
    else:
        val_auc, fpr, tpr = float("nan"), None, None

    if log_images and len(wandb_images) > 0:
        wandb.log({"val_gradcam_analysis": wandb_images}, step=step)

    # Restituisce anche F1 e Precision
    return val_loss, val_acc, val_auc, val_f1, val_precision, (fpr, tpr)

def training(model, train_loader, val_loader, optimizer, lr, device, num_epochs, save_dir="./checkpoints", use_amp=True, images_per_epoch=20, selected_model='multimodal'):
    """
    Standard training loop with Metrics calculation in training phase.
    """
    os.makedirs(save_dir, exist_ok=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        poly_lr_scheduler(optimizer, lr, epoch)
        print('Starting Training Epoch', epoch)
        model.train()
        
        train_losses = []
        # --- MODIFICA: Liste per accumulare predizioni training ---
        train_preds_list = []
        train_labels_list = []

        for batch in train_loader:
            if selected_model == 'MLP_tabular':
                inputs = {"tabular": batch["tabular"].to(device)}
            elif selected_model == 'multi_CNN':
                inputs = {"axial": batch["axial"].to(device), "sagittal": batch["sagittal"].to(device), "coronal": batch["coronal"].to(device)}
            elif selected_model == 'multimodal':
                inputs = {"axial": batch["axial"].to(device), "sagittal": batch["sagittal"].to(device), "coronal": batch["coronal"].to(device), "tabular": batch["tabular"].to(device)}
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            
            # --- MODIFICA: Calcolo probs e preds per metriche training ---
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            train_preds_list.append(preds.detach().cpu())
            train_labels_list.append(labels.detach().cpu())

        # Calcolo metriche Training a fine epoca
        train_loss = float(np.mean(train_losses)) if len(train_losses) else 0.0
        
        train_all_preds = torch.cat(train_preds_list).numpy().astype(int)
        train_all_labels = torch.cat(train_labels_list).numpy().astype(int)
        
        train_acc = accuracy_score(train_all_labels, train_all_preds)
        train_f1 = f1_score(train_all_labels, train_all_preds, average='binary')
        train_prec = precision_score(train_all_labels, train_all_preds, average='binary', zero_division=0)

        # Validation Step (restituisce più metriche ora)
        val_loss, val_acc, val_auc, val_f1, val_prec, (fpr, tpr) = validation(
            model=model, val_loader=val_loader, device=device, criterion=criterion, 
            selected_model=selected_model, log_images=True, images_per_epoch=images_per_epoch, step=epoch
        )

        # --- MODIFICA WANDB: Raggruppamento metriche con "/" ---
        # Usando "Metrica/Train" e "Metrica/Val", WandB li metterà automaticamente
        # sullo stesso grafico nella dashboard predefinita.
        log_dict = {
            "epoch": epoch,
            "Loss/Train": train_loss,
            "Loss/Val": val_loss,
            "Accuracy/Train": train_acc,
            "Accuracy/Val": val_acc,
            "F1_Score/Train": train_f1,
            "F1_Score/Val": val_f1,
            "Precision/Train": train_prec,
            "Precision/Val": val_prec,
            "AUC/Val": val_auc
        }
        wandb.log(log_dict, step=epoch)

        print(f"Epoch [{epoch:03d}/{num_epochs}]")
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}   | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(save_dir, "best_model.pth")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_val_acc": best_val_acc}, ckpt_path)
            wandb.save(ckpt_path)
            wandb.run.summary["best_val_acc"] = best_val_acc
            print(f" Saved best model (val_acc={best_val_acc:.4f})")

    return best_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--excel_path", type=str, required=True)
    parser.add_argument("--selected_model", type=str, required=True)
    parser.add_argument("--split_strategy", type=str, required=True) 
    parser.add_argument("--label_col", type=str, default="label2")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--wandb_project", type=str, default="multiview-vessel")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--test_excel_path", type=str, required=False)
    parser.add_argument("--root_dir_test", type=str, required=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    # Load Training Data
    df = pd.read_excel(args.excel_path)
    ids_all = df["file_sorgente"].astype(str).values
    patient_ids_all = np.array([s.rsplit("_", 1)[0] for s in ids_all])
    idx_all = np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_rows, va_rows = next(gss.split(idx_all, groups=patient_ids_all))
    
    drop_cols = {"file_sorgente", "label1", "label2"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    tab_cols = [c for c in numeric_cols if c not in drop_cols]
    
    scaler = StandardScaler()
    scaler.fit(df.iloc[tr_rows][tab_cols].astype(np.float32).values)
    
    train_transform, val_transform = get_transforms()
    
    train_dataset_full = MRAVesselMultiViewDataset(
        root_dir=args.root_dir,
        excel_path=args.excel_path,
        label_col=args.label_col,
        tabular_cols=tab_cols,
        tabular_scaler=scaler,
        drop_cols=list(drop_cols),
        transform=train_transform
    )

    val_dataset_full = MRAVesselMultiViewDataset(
        root_dir=args.root_dir,
        excel_path=args.excel_path,
        label_col=args.label_col,
        tabular_cols=tab_cols,
        tabular_scaler=scaler,
        drop_cols=list(drop_cols),
        transform=val_transform
    )
    
    if args.split_strategy == 'group_wise':
        train_ds = Subset(train_dataset_full, tr_rows)
        val_ds = Subset(val_dataset_full, va_rows)
    else:
        dataset_len = len(train_dataset_full)
        n_val = int(dataset_len * args.val_ratio)
        n_train = dataset_len - n_val
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(dataset_len, generator=generator).tolist()
        train_ds = Subset(train_dataset_full, indices[:n_train])
        val_ds = Subset(val_dataset_full, indices[n_train:])    

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Init
    if args.selected_model == 'multi_CNN':
        model = MultiViewResNet(args.backbone, pretrained=True, hidden_dim=args.hidden_dim).to(device)
    elif args.selected_model == 'MLP_tabular':
        model = MLP_tabular(train_dataset_full.tabular_dim, 64, 128, args.hidden_dim, 0.5).to(device)
    elif args.selected_model == 'multimodal':
        model = MultiModalMultiViewResNet(train_dataset_full.tabular_dim, args.backbone, True, 64, 128, args.hidden_dim, 0.5).to(device)
    else:
        print('Not supported model'); return

    optimizer = select_optimizer(args, model)
    
    # Training
    best_val_acc = training(model, train_loader, val_loader, optimizer, args.lr, device, args.epochs, "./checkpoints_multiview", selected_model=args.selected_model)
    print("Best val accuracy:", best_val_acc)
    wandb.finish()
    
    # Test
    if args.test_excel_path is not None:
        test_run_name = args.wandb_run_name + "_TEST" if args.wandb_run_name else None
        wandb.init(project=args.wandb_project, name=test_run_name, config=vars(args))
        print("Running test pipeline...")
        test_acc, test_auc = run_test_pipeline(
            args=args,
            model_instance=model, 
            save_results=True,
            output_csv=args.wandb_project + "_predictions_TestSet.csv",
            log_images_to_wandb=True 
        )
        wandb.finish()

if __name__ == "__main__":
    main()