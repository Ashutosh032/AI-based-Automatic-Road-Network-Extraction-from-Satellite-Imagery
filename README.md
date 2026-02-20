# AI-based-Automatic-Road-Network-Extraction-from-Satellite-Image

1. Repository Structure
Plaintext
├── data/               # Documentation on datasets (SpaceNet, DeepGlobe)
├── models/             # Model architectures (U-Net, DeepLabV3+, D-LinkNet)
├── notebooks/          # Exploratory Data Analysis (EDA) and Training logs
├── src/                # Core processing scripts
│   ├── preprocess.py   # Satellite image tiling and normalization
│   ├── train.py        # Training loops
│   ├── inference.py    # Running model on new satellite imagery
│   └── postprocess.py  # Converting pixel masks to Vector (GeoJSON/SHP)
├── weights/            # Saved model checkpoints (.pth or .h5)
├── requirements.txt    # Python dependencies
└── README.md           # Project overview and setup instructions

Description
This project focuses on the automated extraction of road networks from high-resolution satellite imagery (0.3m - 1m resolution). Using Deep Learning, the system performs semantic segmentation to identify road pixels and classifies road surface materials (Bituminous, Concrete, Earthen).

Key Features

Semantic Segmentation: Utilizing U-Net and DeepLabV3+ architectures for pixel-level road detection.

Material Classification: Multi-class classification to distinguish between asphalt, concrete, and unpaved roads.

Connectivity Preservation: Implementation of specialized loss functions (like Soft-IoU or Dice Loss) to ensure thin road structures remain connected.

Vectorization: Automated post-processing pipeline to convert binary masks into clean GIS-compatible vector lines.

Technology Stack

Frameworks: PyTorch / TensorFlow

Computer Vision: OpenCV, Albumentations (for heavy satellite data augmentation)

Geospatial Tools: GDAL, Rasterio, Geopandas (for handling .tif and .shp files)

Models: U-Net, D-LinkNet (optimized for road extraction)

Dataset References
To replicate or extend this work, the following open-source datasets are recommended:

SpaceNet (Road Detection): High-resolution satellite imagery with labeled road vectors.

DeepGlobe Road Extraction Challenge: Large-scale dataset for binary road segmentation.

RoadNet-V2: Multi-scale dataset focused on urban and rural connectivity.

How it Works

Preprocessing: Satellite images are tiled into 512x512 patches and normalized.

Training: The model learns to ignore occlusions (tree canopies/shadows) using multi-spectral data.

Inference: The model outputs a probability map of road locations.

Vectorization: The probability map is skeletonized to produce a center-line road network.
