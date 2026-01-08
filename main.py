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
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, random_split,  Dataset, Subset
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from models.multi_resnet import MultiViewResNet
from models.multi_modal_resnet import MultiModalMultiViewResNet
from models.mlp import MLP_tabular
from preprocess.mra_processing import MRAVesselMultiViewDataset
from utils import denorm_to_uint8, make_triplet_figure, log_val_images_to_wandb, poly_lr_scheduler, select_optimizer, select_splitting_strategy


def validation(model, val_loader, device, criterion, selected_model, log_images=True, images_per_epoch=20, step=None):
    """
    Run validation for one epoch.

    Computes:
    - average validation loss
    - accuracy
    - AUC + ROC curve (only if both classes are present in y_true)
    Log images to wandb representing the three views

    Parameters
    ----------
    model : torch.nn.Module
        Multi-view network under evaluation.
    val_loader : torch.utils.data.DataLoader
        Loader serving validation batches.
    device : torch.device
        Device where inference is run.
    criterion : torch.nn.Module
        Loss function (e.g., BCEWithLogitsLoss).
    selected_model : str
        Selected model type among {'multi_CNN', 'MLP_tabular', 'multimodal'}.
    log_images : bool, optional
        If True, log example triplets from the first validation batch.
    images_per_epoch : int, optional
        Maximum number of triplets to log for the epoch.
    step : int, optional
        wandb step index (usually epoch).

    Returns
    -------
    tuple
        (val_loss, val_acc, val_auc, (fpr, tpr), y_true, y_pred, y_prob)
    """
    print('Starting Validation')

    # Switch to eval mode
    model.eval()

    losses = []
    y_true = []
    y_pred = []
    y_prob = []

    # We'll store one batch for visualization (only the first one).
    first_batch_for_images = None
    first_batch_probs = None
    first_batch_preds = None


    with torch.no_grad():
        for batch in val_loader:
            # Build the model input dict for multi-view inference.
            if selected_model == 'MLP_tabular':
                inputs = {
                    "tabular": batch["tabular"].to(device),
                }
            elif selected_model == 'multi_CNN':
                inputs = {
                    "axial": batch["axial"].to(device),
                    "sagittal": batch["sagittal"].to(device),
                    "coronal": batch["coronal"].to(device),
                }
            elif selected_model == 'multimodal':
                inputs = {
                    "axial": batch["axial"].to(device),
                    "sagittal": batch["sagittal"].to(device),
                    "coronal": batch["coronal"].to(device),
                    "tabular": batch["tabular"].to(device),
                }
            else:
                raise ValueError(f"Selected model '{selected_model}' not supported.")
            labels = batch["label"].to(device)

            # Forward pass
            logits = model(inputs)

            # Compute validation loss on logits
            loss = criterion(logits, labels)
            # Apply sigmoid to get probabilities.
            probs = torch.sigmoid(logits)
            # Threshold probabilities (0/1)
            preds = (probs >= 0.5).float()


            losses.append(loss.item())
            y_true.append(labels.detach().cpu())
            y_pred.append(preds.detach().cpu())
            y_prob.append(probs.detach().cpu())

            # Save the first batch for logging images later (once per epoch).
            if log_images and (first_batch_for_images is None):
                first_batch_for_images = batch
                first_batch_probs = probs.detach().cpu()
                first_batch_preds = preds.detach().cpu()


    y_true = torch.cat(y_true).numpy().astype(int)
    y_pred = torch.cat(y_pred).numpy().astype(int)
    y_prob = torch.cat(y_prob).numpy()

    # Aggregate scalar metrics.
    val_loss = float(np.mean(losses)) if len(losses) else 0.0
    val_acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0

    # AUC + ROC are only defined if both classes appear in y_true.
    if len(np.unique(y_true)) == 2:
        val_auc = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    else:
        val_auc = float("nan")
        fpr, tpr = None, None

    # Log images to wandb (see in utils.py)
    if log_images and (first_batch_for_images is not None) and (images_per_epoch > 0):
        log_val_images_to_wandb(
            batch=first_batch_for_images,
            probs=first_batch_probs,  
            preds=first_batch_preds,  
            max_items=images_per_epoch,
            step=step
        )

    return val_loss, val_acc, val_auc, (fpr, tpr), y_true, y_pred, y_prob



def training(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr,
    device,
    num_epochs,
    save_dir="./checkpoints",
    use_amp=True,
    images_per_epoch=20,
    selected_model='multimodal',
):
    """
    Train a multi-view model validate each epoch.


    Parameters
    ----------
    model : torch.nn.Module
        Network to optimize.
    train_loader, val_loader : torch.utils.data.DataLoader
        Training and validation dataloaders.
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters.
    lr : float
        Initial/base learning rate used by the polynomial scheduler.
    device : torch.device
        Target device for tensors and model.
    num_epochs : int
        Number of training epochs.
    save_dir : str, optional
        Directory to store checkpoints.
    use_amp : bool, optional
        Enable AMP on CUDA for faster training and lower memory usage.
    images_per_epoch : int, optional
        Number of validation images (three views) to log via wandb each epoch.
    selected_model : str, optional
        Selected model type among {'multi_CNN', 'MLP_tabular', 'multimodal'}.

    Returns
    -------
    float
        Best validation accuracy observed.
    """
    # Ensure checkpoint directory exists.
    os.makedirs(save_dir, exist_ok=True)

    # Loss function setup
    criterion = torch.nn.BCEWithLogitsLoss()

    # Enable AMP only when CUDA is available
    amp_enabled = bool(use_amp and device.type == "cuda")

    # GradScaler prevents underflow in float16 gradients when AMP is enabled.
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        # Update learning rate by using polynomial schedule.
        poly_lr_scheduler(optimizer, lr, epoch)
        
        print('Starting Training Epoch', epoch)
        # Training phase
        model.train()
        train_losses = []

        for batch in train_loader:
            # Move batch to device and build multi-view inputs (see dataset structure in MRAVesselMultiViewDataset)
            if selected_model == 'MLP_tabular':
                inputs = {
                    "tabular": batch["tabular"].to(device),
                }
            elif selected_model == 'multi_CNN':
                inputs = {
                    "axial": batch["axial"].to(device),
                    "sagittal": batch["sagittal"].to(device),
                    "coronal": batch["coronal"].to(device),
                }
            elif selected_model == 'multimodal':
                inputs = {
                    "axial": batch["axial"].to(device),
                    "sagittal": batch["sagittal"].to(device),
                    "coronal": batch["coronal"].to(device),
                    "tabular": batch["tabular"].to(device),
                }
            else:
                raise ValueError(f"Selected model '{selected_model}' not supported.")
            labels = batch["label"].to(device)

            # Clear previous gradients.
            optimizer.zero_grad()

            # Autocast runs some ops in float16/bfloat16 for speed on GPU.
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(inputs)
                loss = criterion(logits, labels)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        # Average training loss for the epoch.
        train_loss = float(np.mean(train_losses)) if len(train_losses) else 0.0

        # Validation phase
        val_loss, val_acc, val_auc, (fpr, tpr), y_true, y_pred, y_prob = validation(
            model=model,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            selected_model=selected_model,
            log_images=True,
            images_per_epoch=images_per_epoch,
            step=epoch,
        )

        # Build a W&B confusion matrix plot.
        cm_plot = wandb.plot.confusion_matrix(
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=["0", "1"],
        )

        # Build a W&B ROC curve plot 
        roc_plot = None
        if fpr is not None and tpr is not None:
            roc_table = wandb.Table(
                data=[[float(a), float(b)] for a, b in zip(fpr, tpr)],
                columns=["fpr", "tpr"]
            )
            roc_plot = wandb.plot.line(roc_table, "fpr", "tpr", title="ROC Curve (val)")

        # Log scalar metrics to W&B for this epoch.
        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/auc": val_auc,
        }
        wandb.log(log_dict, step=epoch)

        print(
            f"Epoch [{epoch:03d}/{num_epochs}] | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_auc={val_auc:.4f}"
        )

        # Save best model checkpoint based on validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(save_dir, "best_model.pth")

            # Save both model and optimizer states for reproducibility/resume.
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            }, ckpt_path)

            # Also store best metric in W&B run summary.
            wandb.run.summary["best_val_acc"] = best_val_acc
            print(f" Saved best model (val_acc={best_val_acc:.4f})")

    return best_val_acc


def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--excel_path", type=str, required=True)
    parser.add_argument("--selected_model", type=str, required=True) # { 'MLP_tabular','multi_CNN', multimodal'}
    parser.add_argument("--split_strategy", type=str, required=True) # {'random', 'group_wise'}
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

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    
    # Load the tabular data to perform group-wise splitting
    df = pd.read_excel(args.excel_path)
    
    # Extract patient IDs (without model suffix) for group-wise splitting
    ids_all = df["file_sorgente"].astype(str).values
    patient_ids_all = np.array([s.rsplit("_", 1)[0] for s in ids_all])
    
    idx_all = np.arange(len(df))
    # Group-wise split: patients in train and val sets do not overlap
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_rows, va_rows = next(gss.split(idx_all, groups=patient_ids_all))
    
    # Determine tabular columns (numeric columns excluding drop_cols)
    drop_cols = {"file_sorgente", "label1", "label2"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    tab_cols = [c for c in numeric_cols if c not in drop_cols]
    # Fit scaler on training tabular data
    scaler = StandardScaler()
    scaler.fit(df.iloc[tr_rows][tab_cols].astype(np.float32).values)
    
    # Prepare dataset and data loaders
    dataset = MRAVesselMultiViewDataset(
        root_dir=args.root_dir,
        excel_path=args.excel_path,
        label_col=args.label_col,
        tabular_cols=tab_cols,
        tabular_scaler=scaler,
        drop_cols=list(drop_cols),
    )
    
    print("Samples of dataset:", len(dataset))
    # Select splitting strategy (random or group-wise)
    train_ds, val_ds = select_splitting_strategy(args, dataset)
    print("Training samples:", len(train_ds))
    print("Validation samples:", len(val_ds))
    # Data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.selected_model == 'multi_CNN':
        # Model and optimizer setup
        model = MultiViewResNet(
            backbone_name=args.backbone,
            pretrained=True,
            hidden_dim=args.hidden_dim,
        ).to(device)
    elif args.selected_model == 'MLP_tabular':
        model = MLP_tabular(
            tabular_dim=dataset.tabular_dim,
            tab_emb_dim=64,
            tab_hidden=128,
            hidden_layer=args.hidden_dim,
            dropout_p=0.5,
        ).to(device)
    elif args.selected_model == 'multimodal':
        model = MultiModalMultiViewResNet(
            tabular_dim=dataset.tabular_dim,
            backbone_name=args.backbone,
            pretrained=True,
            tab_emb_dim=64,
            tab_hidden=128,
            fusion_hidden=args.hidden_dim,
            dropout_p=0.5,
        ).to(device)
    else:
        print('Not supported model \n')
        return


    optimizer = select_optimizer(args, model)
    # Start training
    best_val_acc = training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr=args.lr,
        device=device,
        num_epochs=args.epochs,
        selected_model=args.selected_model,
        save_dir="./checkpoints_multiview",
    )

    print("Best val accuracy:", best_val_acc)
    wandb.finish()


if __name__ == "__main__":
    main()