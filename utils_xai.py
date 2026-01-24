# utils_xai.py (Nuovo File)
import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class MultiViewGradCAM:
    def __init__(self, model, device):

        self.model = model
        self.device = device
        # Define target layers for each backbone based on the architecture

        self.target_layers_axial = [model.backbone_axial.layer4[-1]]
        self.target_layers_sagittal = [model.backbone_sagittal.layer4[-1]]
        self.target_layers_coronal = [model.backbone_coronal.layer4[-1]]

        # Initialize GradCAM for each view
        self.cam_axial = GradCAM(model=model, target_layers=self.target_layers_axial)
        self.cam_sagittal = GradCAM(model=model, target_layers=self.target_layers_sagittal)
        self.cam_coronal = GradCAM(model=model, target_layers=self.target_layers_coronal)

    def generate_maps(self, input_tensor_dict, target_category=None):
        """
        Generate heatmaps for a single sample (or batch=1).
        input_tensor_dict: dict with keys 'axial', 'sagittal', 'coronal' and values as tensors [1, 3, H, W]
        target_category: 0 or 1 (if None, use the highest predicted class)
        """

        # If target_category is None, GradCAM will use the predicted class
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None

        grayscale_cam_ax = self.cam_axial(input_tensor=input_tensor_dict, targets=targets)
        grayscale_cam_sag = self.cam_sagittal(input_tensor=input_tensor_dict, targets=targets)
        grayscale_cam_cor = self.cam_coronal(input_tensor=input_tensor_dict, targets=targets)

        return grayscale_cam_ax[0, :], grayscale_cam_sag[0, :], grayscale_cam_cor[0, :]

def overlay_heatmap(original_img_tensor, cam_mask):
    """
    Overlay the CAM heatmap on the original image tensor.
    Parameters
    ----------
    original_img_tensor: torch.Tensor of shape [3, H, W], values normalized as in training
    cam_mask: numpy array of shape [H, W], values in [0, 1]
    
    Returns a numpy array of the overlayed image. 
    """
    # Denormalize the image tensor
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = original_img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Overlay the heatmap
    visualization = show_cam_on_image(img, cam_mask, use_rgb=True)
    return visualization