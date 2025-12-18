import os
import pandas as pd
from PIL import Image

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
                "id":        string identifier
            }
    """

    def __init__(
        self,
        root_dir: str,
        excel_path: str,
        label_col: str = "label2",
        transform=None,
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
        transform : callable, optional
            Transform applied to every loaded image.
        """
        self.root_dir = root_dir
        self.excel_path = excel_path
        self.label_col = label_col

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

        # Pre-compute the list of usable triplets of views
        self.samples = self._build_samples()

    def _build_samples(self):
        """
        Returns
        -------
        list[dict]
            Each entry contains the subject id, the three view paths, and label.
        """
        samples = []
        # Iterate through every subject described in the spreadsheet
        for _, row in self.df.iterrows():
            file_id = str(row["file_sorgente"]).strip()  # e.g., "Normal063-MRA_MANUAL"
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
            Dict containing the three views, the label tensor, and the id.
        """
        sample_info = self.samples[idx]
        paths = sample_info["paths"]
        label = sample_info["label"]

        # Load and transform the three projections for this subject
        axial_img = self._load_image(paths["axial"])
        sagittal_img = self._load_image(paths["sagittal"])
        coronal_img = self._load_image(paths["coronal"])

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return {
            "axial": axial_img,
            "sagittal": sagittal_img,
            "coronal": coronal_img,
            "label": label_tensor,
            "id": sample_info["id"],  # e.g., "Normal063-MRA_MANUAL"
        }
