# VesselVerse Quality Control



## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Project Structure](#project-structure)


## Prerequisites

- Python 3.9+
- PyTorch 2.0+ 
- Git
- Weights & Biases account (optional but recommended)


## Installation

## Project Structure
```
VesselVerse-QualityControl/
├── models/
│   ├── mlp.py                 # MLP Tabuler architecture
│   ├── multi_resnet.py        # Multi-view ResNet architecture 
│   ├── multi_modal_resnet.py  # MultiModal Model (ResNet+MLP) architecture 
│   └── model_ML.py            # Traditional ML baselines and plotting helpers
├── preprocess/
│   ├── mra_processing.py      # Multi-view dataset processing for TIFF projections
│   └── data_preprocessing.py  # Tabular feature processing + VesselVerseProcessing
├── visualization/
│   └── max_intensity_proj.py  # MIP generation from NIfTI volumes
├── utils.py                   # Function useful for training the model
├── main.py                    # Training/validation loop with W&B logging
├── README.md
└── requirements.txt
```