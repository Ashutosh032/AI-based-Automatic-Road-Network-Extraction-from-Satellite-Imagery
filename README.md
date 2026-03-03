# DeepGlobe Road Extraction Project

This project implements a U-Net based model to extract road networks from satellite imagery.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Data**:
    You need the DeepGlobe Road Extraction Dataset.
    - **Option A (Kaggle API)**:
      If you have `kaggle` installed and configured:
      ```bash
      kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset -p data/raw
      unzip data/raw/deepglobe-road-extraction-dataset.zip -d data/raw
      ```
    - **Option B (Manual)**:
      1. Go to [https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
      2. Download the zip file.
      3. Extract the contents into `data/raw/` folder in this project.
      
      Structure should look like:
      ```text
      data/raw/
          train/
              100034_sat.jpg
              100034_mask.png
              ...
      ```

## Training

To train the model:

```bash
cd src
python train.py --data_dir ../data/raw/train --epochs 20 --batch_size 8
```

## Inference

To run inference on a single image:

```bash
cd src
python inference.py --input_image ../data/raw/train/100034_sat.jpg --weights best_model.pth
```
