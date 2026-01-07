import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MRAVesselMultiViewDataset(Dataset):
    """
    Dataset that:
      - loads the axial, sagittal, and coronal MIP views for each subject
      - retrieves the label specified by `file_sorgente` inside the Excel sheet
      - returns a dict:
            {
                "axial":     tensor [3, 224, 224],
                "sagittal":  tensor [3, 224, 224],
                "coronal":   tensor [3, 224, 224],
                "label":     tensor scalar (0.0/1.0),
                "id":        string identifier  (e.g., "Normal063-MRA_MANUAL"),
                "tabular":   tensor [num_tabular_features],
            }
    """

    def __init__(
        self,
        root_dir: str,
        excel_path: str,
        label_col: str = "label2",
        id_col: str = "file_sorgente",
        transform=None,
        tabular_cols=None,
        tabular_scaler=None,
        drop_cols=None,
    ):
        """
        Parameters
        ----------
        root_dir : str
            Root folder containing subdirectories axial/, sagittal/, coronal/.
        excel_path : str
            Path to the Excel file that lists each subject and its label.
        label_col : str, optional
            Name of the column in the Excel sheet that stores the target.
        id_col : str, optional
            Name of the column in the Excel sheet that stores the subject identifier.
        transform : callable, optional
            Transform applied to every loaded image.
        tabular_cols : list[str], optional
            List of column names to use as tabular features. If None, all numeric
            columns except those in `drop_cols` are used.
        tabular_scaler : sklearn.preprocessing object, optional
            Scaler object (e.g., StandardScaler) fitted on the training tabular data.
            If provided, it is used to transform the tabular features.
        drop_cols : list[str], optional
            List of column names to exclude from tabular features.
        """
        self.root_dir = root_dir
        self.excel_path = excel_path
        self.label_col = label_col
        self.id_col = id_col
        self.tabular_cols = tabular_cols
        self.drop_cols = drop_cols
        self.tabular_scaler = tabular_scaler

        # Load the metadata Excel file once to avoid repeated IO
        self.df = pd.read_excel(self.excel_path)

        if transform is None:
            # Default preprocessing (like ImageNet normalization for ResNet)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transform

        if drop_cols is None:
            drop_cols = [self.id_col, "label1", "label2"]
        self.drop_cols = set(drop_cols)
        
        # Prepare tabular data
        self._prepare_tabular()

        # Pre-compute the list of usable triplets of views
        self.samples = self._build_samples()

    def _prepare_tabular(self):
        """
        Prepares the tabular data by selecting specified columns, converting to float32,
        and applying scaling if a scaler is provided.
        """
        if self.tabular_cols is not None:
            tab_df = self.df[self.tabular_cols].copy()
        else:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in self.drop_cols]
            tab_df = self.df[numeric_cols].copy()

        tab_df = tab_df.astype(np.float32)
        tab_arr = tab_df.values

        if self.tabular_scaler is not None:
            tab_arr = self.tabular_scaler.transform(tab_arr)

        self.tabular_df = tab_df
        self.tabular_arr = tab_arr.astype(np.float32)
        self.tabular_dim = self.tabular_arr.shape[1]


    def _build_samples(self):
        """
        Returns
        -------
        list[dict]
            Each entry contains the subject id, the three view paths, and label.
        """
        samples = []
        # Iterate through every subject described in the spreadsheet
        for idx, row in self.df.iterrows():
            file_id = str(row[self.id_col]).strip()  # e.g., "Normal063-MRA_MANUAL"
            label = float(row[self.label_col])

            # Split the identifier into the base study name and the model suffix
            # Examples:
            #   "Normal063-MRA_MANUAL" -> base="Normal063-MRA", suffix="MANUAL"
            #   "Normal063-MRA_nnUNet" -> base="Normal063-MRA", suffix="nnUNet"
            try:
                base, suffix = file_id.rsplit("_", 1)
            except ValueError:
                # Skip rows with unexpected identifiers (no suffix present)
                print(f"Unexpected file_sorgente format: {file_id}")
                continue

            # Build the three filenames with the correct suffix
            # Examples:
            #   base="Normal063-MRA", suffix="MANUAL" -> "Normal063-MRA_mip_axial_MANUAL.tif"
            #   base="Normal063-MRA", suffix="nnUNet" -> "Normal063-MRA_mip_axial_nnUNet.tif"
            filenames = {
                "axial":    f"{base}_mip_axial_{suffix}.tif",
                "sagittal": f"{base}_mip_sagittal_{suffix}.tif",
                "coronal":  f"{base}_mip_coronal_{suffix}.tif",
            }

            paths = {
                view: os.path.join(self.root_dir, view, fname) for view, fname in filenames.items()
            }

            # Keep only entries for which all three projections exist on disk
            if all(os.path.exists(p) for p in paths.values()):
                samples.append(
                    {
                        "row_idx": idx, 
                        "id": file_id,   # includes the suffix so we know the model source
                        "paths": paths,
                        "label": label,
                    }
                )
            else:
                print(f"Missing views for {file_id}: {[v for v,p in paths.items() if not os.path.exists(p)]}")
                pass

        return samples

    def __len__(self):
        """
        Returns
        -------
        int
            Number of valid subjects that have all three projections available.
        """
        return len(self.samples)

    def _load_image(self, path):
        """
        Parameters
        ----------
        path : str
            Absolute path to a TIFF file.

        Returns
        -------
        torch.Tensor
            Image tensor after RGB conversion and transforms.
        """
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.transform(img)
        return img

    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx : int
            Index of the subject inside the samples list.

        Returns
        -------
        dict[str, torch.Tensor | str]
            Dict containing the three views, the label tensor, and the id string. 
        """
        sample_info = self.samples[idx]
        paths = sample_info["paths"]
        label = sample_info["label"]

        # Load and transform the three projections for this subject
        axial_img = self._load_image(paths["axial"])
        sagittal_img = self._load_image(paths["sagittal"])
        coronal_img = self._load_image(paths["coronal"])

        label_tensor = torch.tensor(label, dtype=torch.float32)

        tab = self.tabular_arr[sample_info["row_idx"]]

        return {
            "axial": axial_img,
            "sagittal": sagittal_img,
            "coronal": coronal_img,
            "label": label_tensor,
            "id": sample_info["id"],  # e.g., "Normal063-MRA_MANUAL",
            "tabular": torch.from_numpy(tab).float()
        }
