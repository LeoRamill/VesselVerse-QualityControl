# VesselVerse Quality Control

This repository hosts a Deep Learning pipeline for the automated quality control of MRA (Magnetic Resonance Angiography) volumes. The project employs a multimodal approach combining Convolutional Neural Networks (Multi-View ResNet) for image analysis and an MLP for tabular data, integrated with Explainable AI (XAI) techniques like GradCAM to visualize areas of interest.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Training](#training)
5. [Testing and Inference](#testing-and-inference)
    - [Single Sample Test](#single-sample-test)
    - [Multiple Sample Test (Batch)](#multiple-sample-test-batch)
6. [Supported Models](#supported-models)

## Prerequisites

The project is developed in Python 3.9+. Key dependencies include:
- PyTorch 2.0+ & Torchvision
- NumPy, Pandas, Scikit-learn
- OpenCV (cv2), Pillow
- NiBabel (for NIfTI file handling)
- Grad-CAM
- Weights & Biases (optional, for logging)

See `requirements.txt` for the complete list.

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/LeoRamill/VesselVerse-QualityControl.git](https://github.com/LeoRamill/VesselVerse-QualityControl.git)
   cd VesselVerse-QualityControl
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```text
VesselVerse-QualityControl/
├── models/
│   ├── mlp.py                 # MLP architecture for tabular data
│   ├── multi_resnet.py        # Multi-view ResNet architecture
│   ├── multi_modal_resnet.py  # Multimodal Model (ResNet + MLP)
│   └── model_ML.py            # Traditional ML Baselines
├── preprocess/
│   ├── mra_processing.py      # Multi-view dataset processing
│   └── data_preprocessing.py  # Tabular feature processing
├── visualization/
│   └── max_intensity_proj.py  # MIP generation from NIfTI volumes
├── utils.py                   # Utility functions for training
├── utils_test.py              # Utility functions for testing and pre-processing
├── utils_xai.py               # Explainable AI functions (GradCAM)
├── main.py                    # Training/Validation loop
├── test.py                    # Script for inference on a single NIfTI file
├── multiple_test.py           # Script for batch inference on a folder
├── VESSEL_METRICS.py          # Vascular metrics calculation (external dependency)
└── requirements.txt
```

## Training
To start training the model, use the `main.py` script. You must provide the path to the data (preprocessed TIFF images or NIfTI files) and the Excel file containing labels and features.

```bash
   python main.py \
  --root_dir "./path/to/images" \
  --excel_path "./path/to/dataset.xlsx" \
  --selected_model "multimodal" \
  --epochs 50 \
  --batch_size 16
   ```

## Testing and Inference
The project offers two main testing modes: analyzing a single NIfTI file or processing a batch of files from a folder. Both scripts generate GradCAM visualizations to interpret the model's prediction.

### Single Sample Test
Use `test.py` to analyze a single `.nii.gz` volume. The script loads the trained model, extracts features, generates projections (MIP), and saves a result image with the GradCAM heatmap overlay.

#### Parameters:

- `--nifti_path`: Path to the input NIfTI file.

- `--checkpoint_path`: Path to the saved model .pth file.

- `--excel_path`: Path to the original Excel file (used to fit the scaler on tabular data).

- `--selected_model`: Model type (default: multimodal).

#### Command: 
```bash
python test.py \
  --nifti_path "/content/sample_volume.nii.gz" \
  --checkpoint_path "/content/model_checkpoint.pth" \
  --excel_path "/content/dataset_originale.xlsx" \
  --selected_model "multimodal"
```
#### Output:

A `.jpg image` (e.g., `TestResult_sample_volume.nii.gz.jpg`) will be generated, displaying axial, sagittal, and coronal views with prediction labels.

### Multiple Sample Test (Batch)

Use `multiple_test.py` to automatically process all `.nii.gz` files within a specific folder. In addition to generating GradCAM images, this script produces a summary Excel file with the results.

#### Parameters:

- `--input_folder`: Folder containing the `.nii.gz` files to test.

- `--output_folder`: Folder where output images and the result Excel will be saved.

- `--checkpoint_path`: Path to the model `.pth` file.

- `--excel_path`: Path to the original Excel file (for scaler fitting).

- `--selected_model`: Model type (default: multimodal).

#### Command:
```bash
python multiple_test.py \
  --input_folder "/content/test_folder_IXI" \
  --output_folder "/content/output_results" \
  --checkpoint_path "/content/model_checkpoint.pth" \
  --excel_path "/content/dataset_originale.xlsx" \
  --selected_model "multimodal"
```
#### Output:

Images: One `.jpg` image for each analyzed volume, saved in the output folder.

Report: A `Results_Summary.xlsx` file containing:

- Filename
- Prediction (0 or 1)
- Probability
- Text Label (Good/Bad)
  
### Resource

You can find the `--excel_path` and `--checkpoint_path` model here:
```text
  https://drive.google.com/drive/folders/1LP31l8sUBt7NQChTy_WMVPtzjOmM00li?usp=sharing
```

## Supported Models
The framework supports different architectures, selectable via the `--selected_model` parameter:

- `multi_CNN`: Network using only images (MIP projections) via a Multi-View ResNet.
- `MLP_tabular`: Network using only tabular features extracted from the volume.
- `multimodal` (Recommended): Hybrid network fusing visual features (CNN) and tabular features (MLP) for more accurate classification.
