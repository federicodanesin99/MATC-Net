# MACTNet: WiFi-based HAR with Dual-Transformer Architecture

## Overview

MACTNet is a neural architecture designed for WiFi-based Human Activity Recognition (HAR). It leverages a dual-transformer encoder mechanism, analyzing Channel State Information (CSI) from both temporal and channel perspectives. This repository contains the code for training, testing, and optionally fine-tuning (transfer learning) the MACTNet model.

## Prerequisites

1. **Python 3.8+** (or similar)
2. **Required Python Libraries**:
   - TensorFlow (>= 2.0)
   - NumPy
   - Matplotlib
   - scikit-learn
   - seaborn

## Setup Instructions

1. **Download** the dataset at https://researchdata.cab.unipd.it/624/ and extract the dataset and organize it inside the HDA_dataset folder as follows
   HDA_dataset/
    ├── AR-1A/
    ├── AR-1B/
    ├── AR-2A/
    ├── AR-2B/
    ├── ...
2. **Install the required Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   or install them individually:
   ```bash
   pip install tensorflow numpy matplotlib scikit-learn seaborn
   ```

## Project Structure

- **`config.py`**: Python-based file containing hyperparameters (batch size, sequence length, etc.), dataset paths, and other settings.
- **`main.py`**: Main training script that trains the MACTNet model using parameters from `config.py`.
- **`model.py`**: Contains the MACTNet class definition and any necessary custom layers or callbacks.
- **`preprocessing.py`**: Functions for data preprocessing, segmentation, normalization, and label encoding.
- **`test_and_transfer.py`**: Script demonstrating how to load a pretrained model, evaluate it on a new dataset, and optionally perform transfer learning.
- **`CSI_phase_sanitization_signal_preprocessing.py`**: (GPL-licensed) script for advanced CSI phase sanitization and signal preprocessing. It includes:
  - Activity mapping (Walking, Running, Jumping, etc.)
  - Optional argument parsing (`argparse`) for directory paths, multi-antenna parameters, etc.
  - Data loading and FFT shifting from `.mat` files
  - Sanitization steps (removing empty frames, normalizing streams, etc.)
  - Output pickling of the preprocessed signals

## How to Use

### 1. Configure Hyperparameters

Edit `config.py` to specify your desired hyperparameters and dataset paths, for example:

```python
batch_size = 32
sequence_length = 200
# ... Other parameters
```

### 2. Preprocess the CSI Data 

If you have raw `.mat` files requiring phase sanitization and advanced processing, you can run:

```bash
python CSI_phase_sanitization_signal_preprocessing.py <dir> <all_dir> <name> <nss> <ncore> <start_idx>

python CSI_phase_sanitization_signal_preprocessing.py HDA_dataset/AR-3a/ 1 - 1 4 0 


```

where:

- `<dir>` is the dataset directory (e.g., `HDA_dataset/AR-1b/`)
- `<all_dir>` is `1` if you want to process all `.mat` files in the folder
- `<name>` is the specific filename to process (if `<all_dir>`=0)
- `<nss>` is the number of spatial streams
- `<ncore>` is the number of cores (for indexing in the raw data)
- `<start_idx>` indicates the starting frame index for data processing

This script is provided under the **GNU General Public License v3 (GPL-3.0)**.

### 3. Train the MACTNet Model

Run the following command to start training:

```bash
python main.py
```

- **What happens**:
  - The script loads configuration parameters from `config.py` (e.g., batch size, number of epochs, model hyperparameters).
  - It preprocesses the CSI data (optionally using the preprocessed outputs from the prior step), splits it into training/validation sets, and creates TensorFlow datasets.
  - It initializes and compiles the MACTNet model.
  - Finally, it trains the model and saves the trained weights to `transformer_pretrained.weights.h5` and the full model to `model-chan.keras`.

### 4. Test or Transfer Learn

To evaluate or fine-tune the model on a new dataset, use:

```bash
python test_and_transfer.py
```

- **What happens**:
  - The script loads the pretrained model from `model.keras`.
  - It evaluates the model on one or more new datasets (e.g., `AR9b`).
  - Optionally, it performs transfer learning by training on a fraction of the new dataset to adapt the model.
  - Results (confusion matrix, classification metrics, etc.) are displayed.

### 5. Interpreting the Results

- **Logs and Metrics**: The training script prints out training/validation accuracy and loss each epoch.
- **Checkpoints**: Model weights are saved for reuse.
- **Confusion Matrix & PCA Visualizations**: Scripts provide confusion matrix and PCA cluster plots for better understanding of classification performance.

## Customization

- **Change Hyperparameters**: Modify `config.py` to adjust batch size, sequence length, overlap, or the number of transformer layers/heads.
- **Change Datasets**: Update dataset paths or names in the same file.
- **Add Activities**: Adjust `activity_labels` and `label_mapping` in `config.py`.

## Troubleshooting

1. **ModuleNotFoundError**: Check you installed all the required Python libraries.
2. **TensorFlow Version Issues**: Make sure your TensorFlow version is compatible with your Python version.
3. **Dataset or Path Errors**: Ensure `config.py` has the correct paths or references to your data.

## License

- **`CSI_phase_sanitization_signal_preprocessing.py`**: Released under the [GNU General Public License v3](https://www.gnu.org/licenses/).

## Contact

For questions or comments, please reach out to the project maintainer:

- **Name**: Federico Danesin
- **Email**: [federicodanesin99@gmail.com](mailto\:federicodanesin99@gmail.com)
