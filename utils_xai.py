# utils_xai.py (Nuovo File)
import numpy as np
import torch
import cv2
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class ValidationModelWrapper(nn.Module):
    """
    Wrapper that adapts the multi-view input format for GradCAM.
    Combines the three views into a single tensor and reconstructs the input
    dictionary for the original model during the forward pass.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tabular_context = None 

    def set_tabular(self, tab_data):
        self.tabular_context = tab_data

    def forward(self, x):
        # x shape: (B, 9, H, W) -> split into three views of 3 channels each
        xa = x[:, 0:3, :, :]
        xs = x[:, 3:6, :, :]
        xc = x[:, 6:9, :, :]
        
        batch = {
            "axial": xa,
            "sagittal": xs,
            "coronal": xc
        }
        
        if self.tabular_context is not None:
            batch["tabular"] = self.tabular_context
                    
        logits = self.model(batch)
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        return logits

class MultiViewGradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.wrapper = ValidationModelWrapper(model)
        self.wrapper.eval()
        
        # Define target layers for each backbone based on the architecture
        self.target_layers_axial = [model.backbone_axial.layer4[-1]]
        self.target_layers_sagittal = [model.backbone_sagittal.layer4[-1]]
        self.target_layers_coronal = [model.backbone_coronal.layer4[-1]]

        # Initialize GradCAM for each view
        self.cam_axial = GradCAM(model=self.wrapper, target_layers=self.target_layers_axial)
        self.cam_sagittal = GradCAM(model=self.wrapper, target_layers=self.target_layers_sagittal)
        self.cam_coronal = GradCAM(model=self.wrapper, target_layers=self.target_layers_coronal)

    def generate_maps(self, input_tensor_dict, target_category=None):
        """
        Generate heatmaps for a single sample (or batch=1).
        input_tensor_dict: dict with keys 'axial', 'sagittal', 'coronal' and values as tensors [1, 3, H, W]
        target_category: 0 or 1 (if None, use the highest predicted class)
        """

        # If target_category is None, GradCAM will use the predicted class
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
        # Extract individual view tensors
        xa = input_tensor_dict["axial"]
        xs = input_tensor_dict["sagittal"]
        xc = input_tensor_dict["coronal"]
        xt = input_tensor_dict.get("tabular", None)
        
        # Set tabular data in the wrapper context
        self.wrapper.set_tabular(xt)
        
        # Create concatenated input tensor
        # (B, 3, H, W) -> (B, 9, H, W)
        input_tensor_concat = torch.cat([xa, xs, xc], dim=1)
        
        # Generate heatmaps by calling GradCAM on the concatenated tensor
        # The wrapper will handle unpacking and passing to the original model
        cam_ax = self.cam_axial(input_tensor=input_tensor_concat, targets=targets)[0, :]
        cam_sag = self.cam_sagittal(input_tensor=input_tensor_concat, targets=targets)[0, :]
        cam_cor = self.cam_coronal(input_tensor=input_tensor_concat, targets=targets)[0, :]

        self.wrapper.set_tabular(None)
        return cam_ax, cam_sag, cam_cor

def overlay_heatmap(tensor_img_norm, cam_mask):
    """Overlay the Grad-CAM heatmap on the normalized tensor image.
    Args:
        tensor_img_norm: Tensor immagine normalizzata (3, H, W) float32 0-1
        cam_mask: La maschera di attivazione (H, W) float 0-1
    Returns:
        visualization: Immagine sovrapposta (H, W, 3) uint8
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor_img_norm.cpu().detach().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return show_cam_on_image(img, cam_mask, use_rgb=True)

def overlay_heatmap_original_size(original_image_np, cam_mask):
    """
    Overlay the Grad-CAM heatmap on the original image size.
    Args:
        original_image_np: Immagine originale (H, W, 3) uint8 o float32 0-1
        cam_mask: La maschera di attivazione (H_cam, W_cam) float 0-1
    Returns:
        visualization: Immagine sovrapposta (H, W, 3) uint8
    """
    # Normalize original image to 0-1 float
    if original_image_np.max() > 1.0:
        img = original_image_np.astype(np.float32) / 255.0
    else:
        img = original_image_np.astype(np.float32)

    # Resize cam_mask to original image size
    h_orig, w_orig = img.shape[:2]
    cam_mask_resized = cv2.resize(cam_mask, (w_orig, h_orig))
    # Overlay heatmap on image
    visualization = show_cam_on_image(img, cam_mask_resized, use_rgb=True)
    
    return visualization