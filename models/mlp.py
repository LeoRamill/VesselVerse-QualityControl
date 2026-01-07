import torch
import torch.nn as nn
from torchvision import models

class MLP_tabular(nn.Module):
    def __init__(
        self,
        tabular_dim: int,
        tab_emb_dim: int = 64,
        tab_hidden: int = 128,
        hidden_layer: int = 256,
        dropout_p: float = 0.5,
    ):
        super().__init__()


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
            nn.Linear(tab_emb_dim, hidden_layer),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_layer, 1),  # logit
        )

    def forward(self, batch):
        xt = batch["tabular"]
        tab_feat = self.tabular_mlp(xt)
        logits = self.classifier(tab_feat).squeeze(-1)
        return logits
