import torch
import torch.nn as nn
from torchvision import models


class MultiViewResNet(nn.Module):
    """
    Multi-view model composed of independent ResNet backbones for axial,
    sagittal, and coronal planes whose embeddings are concatenated and fed
    to an MLP for binary classification.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Build three independent backbones
        self.backbone_axial, feat_dim = self._build_backbone(backbone_name, pretrained)
        self.backbone_sagittal, _ = self._build_backbone(backbone_name, pretrained)
        self.backbone_coronal, _ = self._build_backbone(backbone_name, pretrained)

        # Final classifier: input = 3 * feat_dim, output = single logit
        self.classifier = nn.Sequential(
            nn.Linear(3 * feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, 1),
        )

    def _build_backbone(self, backbone_name, pretrained):
        """
        Build a ResNet without the last fully connected layer, returning its embedding.

        Parameters
        ----------
        backbone_name : str
            Name of the torchvision ResNet backbone to load.
        pretrained : bool
            Whether to initialize the backbone with ImageNet weights.

        Returns
        -------
        torch.nn.Module
            ResNet backbone with the final fully connected layer replaced by Identity.
        int
            Dimension of the resulting embedding.
        """
        if backbone_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif backbone_name == "resnet34":
            model = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone '{backbone_name}' not supported in this example.")

        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feat_dim

    def forward(self, x):
        """
        Execute the three-view ensemble and return one logit per subject.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Mapping with the keys "axial", "sagittal", and "coronal";
            each tensor has shape [B, 3, H, W].

        Returns
        -------
        torch.Tensor
            Vector of logits with shape [B]; values are pre-sigmoid.
        """
        # Split the input dict into the three views
        x_axial = x["axial"]
        x_sagittal = x["sagittal"]
        x_coronal = x["coronal"]

        # Encode each plane independently using its dedicated ResNet backbone
        feat_axial = self.backbone_axial(x_axial)
        feat_sagittal = self.backbone_sagittal(x_sagittal)
        feat_coronal = self.backbone_coronal(x_coronal)

        # Fuse the three latent vectors so the classifier can see all planes at once
        feat_concat = torch.cat(
            [feat_axial, feat_sagittal, feat_coronal],
            dim=1,
        )

        # Return a single logit per sample; caller can apply sigmoid if needed
        logits = self.classifier(feat_concat).squeeze(-1)
        return logits
