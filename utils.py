import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
from torch.optim import SGD, Adam, RMSprop

def select_optimizer(args, model):
    """
    Choose optimizers: RMSprop, SGD, Adam.

    Parameters
    ----------
        args: argparse.Namespace
            Must expose optimizer choice plus LR, momentum, and weight_decay values.
        model: torch.nn.Module
            Model whose parameters will be optimized.

    Returns
    -------
        torch.optim.Optimizer | None
            Configured optimizer instance, or None if the choice is unsupported.
    """
    if args.optimizer == 'rmsprop':
        optimizer = RMSprop(model.parameters(), args.lr)
    elif args.optimizer == 'sgd':
        optimizer = SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:  
        print('not supported optimizer \n')
        return None
    return optimizer


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=50, power=0.9):
    """
    Polynomial learning-rate schedule.

    LR is updated as:
        lr = init_lr * (1 - iter / max_iter) ** power

    Parameters
    ----------
        optimizer (torch.optim.Optimizer): Optimizer whose LR to update.
        init_lr (float): Base learning rate at the start.
        iter (int): Current iteration index (here used as epoch index).
        lr_decay_iter (int): Unused in this implementation.
        max_iter (int): Iteration index at which lr approaches 0.
        power (float): Polynomial exponent controlling decay shape.

    Returns
    -------
        float: The updated learning rate.
    """
    # Compute decayed learning rate as a scalar float.
    lr = init_lr * (1 - iter / max_iter) ** power

    # Update the optimizer's learning rate.
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

    return lr

def denorm_to_uint8(img_tensor):
    """
    Convert a normalized torch image tensor to a uint8 RGB image.

    Useful for visualization because training pipelines often store images in normalized float format.

    Parameters
    ----------
    img_tensor : torch.Tensor
        Tensor with shape [3, H, W] normalized as in typical ImageNet preprocessing.

    Returns
    -------
    numpy.ndarray
        uint8 image with shape [H, W, 3] in RGB, values in [0, 255].
    """
    
    t = img_tensor.detach().cpu().clone()

    # Normalization like in preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    t = t * std + mean

    # Clamp to valid display range [0, 1] before converting to 8-bit.
    t = torch.clamp(t, 0, 1)

    # Convert from [3, H, W] to [H, W, 3] and scale to [0, 255] uint8.
    t = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return t


def make_triplet_figure(axial_u8, sagittal_u8, coronal_u8, title=""):
    """
    Create a horizontal montage figure with three subplots: Axial, Sagittal, Coronal.

    Parameters
    ----------
    axial_u8, sagittal_u8, coronal_u8 : numpy.ndarray
        uint8 images (H, W, 3) for each plane.
    title : str, optional
        Title shown above the montage.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib Figure with 3 subplots (Axial / Sagittal / Coronal).
    """
    fig = plt.figure(figsize=(12, 4))
    fig.suptitle(title)

    gs = fig.add_gridspec(1, 3)

    # Axial view
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(axial_u8)
    ax.set_title("Axial")
    ax.axis("off")

    # Sagittal view
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(sagittal_u8)
    ax.set_title("Sagittal")
    ax.axis("off")

    # Coronal view
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(coronal_u8)
    ax.set_title("Coronal")
    ax.axis("off")

    fig.tight_layout()
    return fig


def log_val_images_to_wandb(batch, probs, preds, max_items=6, step=None):
    """
    Log a set of validation triplets to Weights & Biases.

    We log up to `max_items` samples from a batch, each sample as a triplet figure
    (axial/sagittal/coronal) with a caption including ID, true label, prediction and probability.

    Parameters
    ----------
    batch : dict
        Must contain: "axial", "sagittal", "coronal", "label", "id".
    probs : torch.Tensor
        Predicted probabilities (same sample ordering as `batch`).
    preds : torch.Tensor
        Binary predictions (0/1 floats or ints), same ordering.
    max_items : int, optional
        Max number of samples to visualize from this batch.
    step : int, optional
        Global step/epoch passed to wandb.log for consistent tracking.
    """
    B = len(batch["id"])
    k = min(B, max_items)

    images = []

    probs_cpu = probs.detach().cpu()
    preds_cpu = preds.detach().cpu()
    labels_cpu = batch["label"].detach().cpu()

    for i in range(k):
        # Convert each image from normalized tensor -> uint8 numpy for plotting.
        axial_u8 = denorm_to_uint8(batch["axial"][i])
        sagittal_u8 = denorm_to_uint8(batch["sagittal"][i])
        coronal_u8 = denorm_to_uint8(batch["coronal"][i])

        # Extract scalar values
        true_label = float(labels_cpu[i].item())
        pred_label = int(preds_cpu[i].item())
        prob = float(probs_cpu[i].item())
        sample_id = batch["id"][i]

        # Caption
        title = f"{sample_id} | true={true_label:.0f} pred={pred_label} prob={prob:.3f}"

        # Matlplotlib figure with 3 views
        fig = make_triplet_figure(axial_u8, sagittal_u8, coronal_u8, title=title)
        images.append(wandb.Image(fig, caption=title))

        plt.close(fig)

    # Log to wandb
    wandb.log({"val/examples": images}, step=step)
    
