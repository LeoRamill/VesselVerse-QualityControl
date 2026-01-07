import torch
import torch.nn as nn
from torchvision import models

class MultiModalMultiViewResNet(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        tab_emb_dim: int = 64,
        tab_hidden: int = 128,
        fusion_hidden: int = 256,
        dropout_p: float = 0.5,
    ):
        super().__init__()

        self.backbone_axial, feat_dim = self._build_backbone(backbone_name, pretrained)
        self.backbone_sagittal, _ = self._build_backbone(backbone_name, pretrained)
        self.backbone_coronal, _ = self._build_backbone(backbone_name, pretrained)

        vision_dim = 3 * feat_dim

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, tab_hidden),
            nn.BatchNorm1d(tab_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=min(0.3, dropout_p)),

            nn.Linear(tab_hidden, tab_emb_dim),
            nn.BatchNorm1d(tab_emb_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(vision_dim + tab_emb_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fusion_hidden, 1),  # logit
        )

    def _build_backbone(self, backbone_name, pretrained):
        if backbone_name == "resnet18":
            m = models.resnet18(pretrained=pretrained)
        elif backbone_name == "resnet34":
            m = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Backbone '{backbone_name}' non supportato in questo snippet.")
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim

    def forward(self, batch):
        xa = batch["axial"]
        xs = batch["sagittal"]
        xc = batch["coronal"]
        xt = batch["tabular"]

        fa = self.backbone_axial(xa)
        fs = self.backbone_sagittal(xs)
        fc = self.backbone_coronal(xc)
        vision_feat = torch.cat([fa, fs, fc], dim=1)

        tab_feat = self.tabular_mlp(xt)

        fused = torch.cat([vision_feat, tab_feat], dim=1)
        logits = self.classifier(fused).squeeze(-1)
        return logits
