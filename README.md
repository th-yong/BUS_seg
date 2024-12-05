# Breast Ultrasound Segmentation Project

## Project Objective
This project utilizes the **MT_small_dataset**, a breast ultrasound dataset available at [Kaggle](https://www.kaggle.com/datasets/mohammedtgadallah/mt-small-dataset). The goal is to:

> 1. Train and compare segmentation models for the "Benign_Original" and "Fuzzy_Benign" datasets.
> 2. Select and utilize an appropriate network for segmentation.
> 3. Split the dataset into training, validation, and test sets in a 7:2:1 ratio.

## Dataset Description
The dataset comprises 1200 breast ultrasound (BUS) images, including benign and malignant cases, along with their respective ground truth segmentation labels. This dataset was originally introduced in the following study:

> Badawy, Samir M., et al. "Automatic semantic segmentation of breast tumors in ultrasound images based on combining fuzzy logic and deep learning—A feasibility study." PLOS ONE, 2021, e0251899.
  
### Dataset Structure
The dataset is divided into folders, but for this project, only the following subsets from the **Benign Folder** are used:

### Benign Folder
- **Original_Benign:** 200 BUS images (128 × 128 × 3) with benign cancer.
- **Fuzzy_Benign:** 200 contrast-enhanced BUS images (128 × 128 × 3) with benign cancer.
- **Ground_Truth_Benign:** 200 ground truth segmentation masks (128 × 128).

### Enhanced Images
The **Fuzzy-enhanced BUS images** were generated using an **FIO-based method** for contrast enhancement. 
<p align="center">
 <img src = "./config/original_Benign plot.png" height="400" width="800">
 <img src = "./config/Fuzzy_Benign plot.png" height="400" width="800">
</p>

### Sample Visualization: Overlay of Images and Ground Truth

To provide a clearer understanding of the dataset and its annotations, we visualized the overlap of the images from **Original_Benign** and **Fuzzy_Benign** with their corresponding segmentation masks from **Ground_Truth_Benign**.

<p align="center">
 <img src="./config/Original_Benign_Overlay.png" alt="Original Benign Overlay" width="800px">
 <br>
 <b>Figure 1:</b> Original_Benign image overlaid with Ground_Truth_Benign.
</p>

<p align="center">
 <img src="./config/Fuzzy_Benign_Overlay.png" alt="Fuzzy Benign Overlay" width="800px">
 <br>
 <b>Figure 2:</b> Fuzzy_Benign image overlaid with Ground_Truth_Benign.
</p>

---

# Results

## Benchmark Networks

| Network          | Framework   | Original Code                                                              | Reference     |
|------------------|-------------|----------------------------------------------------------------------------|---------------|
| **U-Net**        | Caffe       | [GitHub](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net)      | MICCAI'15     |
| **Attention U-Net** | PyTorch  | [GitHub](https://github.com/ozan-oktay/Attention-Gated-Networks)           | Arxiv'18      |
| **U-Net++**      | PyTorch     | [GitHub](https://github.com/4uiiurz1/pytorch-nested-unet)                  | MICCAI'18     |
| **SegResNet**    | PyTorch     | [GitHub](https://github.com/yassouali/pytorch-segmentation/tree/master)    | MICCAI'21     |
| **CMUNeXt**      | PyTorch     | [GitHub](https://github.com/FengheTan9/CMUNeXt)                            | ISBI'24       |

---

## Training and Evaluation

### 1. **Objective**
To evaluate the performance of each benchmark network on the **Original** and **Fuzzy** datasets using the following metrics:
- **Dice Score**: Overlap between prediction and ground truth.
- **Precision**: Ratio of correctly predicted positive observations to total predicted positives.
- **Recall**: Ratio of correctly predicted positive observations to all actual positives.
- **F1 Score**: Harmonic mean of precision and recall.
- **Inference Time**: Time taken to process a single image during testing.

### 2. **Training Details**
- **Dataset**: MT_small_dataset / Benign
- **Training/Validation/Test Split**: 7:2:1
- **Optimizer**: Adam
- **Learning Rate**: $1 \times 10^{-4}$
- **Loss Functions**: Dice Loss
    - **Dice Loss** was chosen as the final loss function for training all networks due to its superior performance
    - For more details, refer to the documentation at `./ablation_study/loss/ReadMe.md`.

### 3. **Model Performance Evaluation**
The evaluation results for each network (on **Original** and **Fuzzy** datasets) will be presented in the following format:

| Network          | Dataset    | Parameters (M) | Dice Score       | Precision        | Recall           | F1 Score         | Inference Time (ms) |
|------------------|------------|----------------|------------------|------------------|------------------|------------------|---------------------|
| U-Net            | Original   | 31.04          | 0.7920 ± 0.1386  | 0.7857 ± 0.2003  | 0.8422 ± 0.1287  | 0.7918 ± 0.1388  | 10.05               |
| U-Net            | Fuzzy      | 31.04          | 0.8141 ± 0.1106  | 0.8391 ± 0.1567  | 0.8206 ± 0.1357  | 0.8139 ± 0.1108  | 10.50               |
| Attention U-Net  | Original   | 34.88          | 0.7531 ± 0.2254  | 0.7280 ± 0.2596  | 0.8431 ± 0.1656  | 0.7528 ± 0.2256  | 11.54               |
| Attention U-Net  | Fuzzy      | 34.88          | 0.7226 ± 0.2041  | 0.7201 ± 0.2871  | 0.8114 ± 0.1234  | 0.7224 ± 0.2043  | 11.60               |
| U-Net++          | Original   | 9.16           | 0.7995 ± 0.1115  | 0.8281 ± 0.1710  | 0.8137 ± 0.1496  | 0.7992 ± 0.1117  | 8.78                |
| U-Net++          | Fuzzy      | 9.16           | 0.8017 ± 0.1107  | 0.8665 ± 0.1153  | 0.7733 ± 0.1706  | 0.8014 ± 0.1109  | 9.38                |
| SegResNet        | Original   | 53.55          | 0.7359 ± 0.1333  | 0.7922 ± 0.1747  | 0.7155 ± 0.1729  | 0.7355 ± 0.1335  | 14.53               |
| SegResNet        | Fuzzy      | 53.55          | 0.7397 ± 0.1583  | 0.7903 ± 0.2034  | 0.7267 ± 0.1817  | 0.7394 ± 0.1586  | 14.32               |
| CMUNeXt          | Original   | 3.15           | 0.7163 ± 0.1949  | 0.7550 ± 0.2481  | 0.7114 ± 0.1602  | 0.7160 ± 0.1952  | 9.60                |
| CMUNeXt          | Fuzzy      | 3.15           | 0.6807 ± 0.3162  | 0.7743 ± 0.3235  | 0.6227 ± 0.3196  | 0.6801 ± 0.3170  | 9.82                |

---

## Installation
Please follow the commands below for more details.

### 1. Clone this repository.
```
git clone https://github.com/th-yong/Breast-Ultrasound
```

### 2. Create conda environment and install required python packages.
```
conda create -n BUS python=3.10
```

### 3. Install packages from requirements.txt.
```
pip install -r requirements.txt
```

---

## Usage

This project allows training and testing segmentation models on the **MT_Small_Dataset**. Depending on your use case, you can choose between the **Original_Benign** or **Fuzzy_Benign** datasets.

### Train

To train the segmentation model, run the following command:

```
python main.py --mode train --dataset original
```

- **Arguments**:
  - `--mode`: Specifies the mode of execution. Use `train` to start training.
  - `--dataset`: Specifies which dataset to use. Options are `original` or `fuzzy`.

### Test

To test a saved segmentation model, run the following command:

```
python main.py --mode test --dataset original --model_path ./results/best_model_original_valloss_0.16.pth
```

- **Arguments**:
  - `--mode`: Specifies the mode of execution. Use `test` to start testing.
  - `--dataset`: Specifies which dataset to use. Options are `original` or `fuzzy`.
  - `--model_path`: Path to the saved model file to evaluate.

