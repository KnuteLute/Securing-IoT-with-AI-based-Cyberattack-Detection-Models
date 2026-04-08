# NorFin Network Intrusion Detection System

This repository accompanies the paper **"Securing IoT with AI-based Cyberattack Detection Models using Real-World Testbeds and Datasets"**.

It contains the preprocessing and TensorFlow training pipeline used to build cyberattack detection models for network traffic classification. Data can be used in two ways:

- via externally prepared raw packet captures (`.pcapng`), or
- via pre-generated CSV feature files if they are included in this repository.

## Overview

The workflow is:

1. Use pre-generated CSV feature files from `NorFin_csv/` (if present), **or** collect `.pcapng` captures.
2. If needed, preprocess the packet captures into windowed CSV feature files.
3. Train TensorFlow models on the CSV data.
4. Review saved model artifacts, metrics, confusion matrices, and feature importance results.

The project targets these traffic categories:

- Benign
- Bruteforce
- Reconnaissance
- Injections
- DoS

## Repository Structure

- `NorFin_Lab/`
  - Expected location for the raw packet captures and trained model outputs.
  - `Attack_Data_Subtype_Labels/` contains the attack subfolders with `.pcapng` files.
  - `benign_data/` contains the benign capture set.
  - `models/` stores trained models, scalers, label encoders, plots, and comparison results.
  - `scripts/` contains the Python code used for preprocessing and training.

- `NorFin_csv/`
  - Generated CSV windows produced from preprocessing.
  - Mirrors the folder structure of the raw capture folders.

## Important Note About Data

This repository may include preprocessed CSV data in `NorFin_csv/`. If CSV files are available, you can train directly without running preprocessing.

For full reproduction from raw traffic, the complete `.pcapng` dataset is still expected to be prepared separately and placed in the expected folder structure.

Expected input structure:

- `NorFin_Lab/Attack_Data_Subtype_Labels/<attack folder>/*.pcapng`
- `NorFin_Lab/benign_data/<benign folder>/*.pcapng`

Expected CSV structure (either generated locally or already provided):

- `NorFin_csv/<attack folder>/*.csv`
- `NorFin_csv/<benign folder>/*.csv`

## Requirements

### Python packages

Install the Python dependencies used by the project:

- tensorflow
- pandas
- scikit-learn
- joblib
- matplotlib
- seaborn
- pyshark

### System dependency

`pyshark` requires **tshark** from Wireshark to be installed and available on PATH.

## Recommended Setup

### 1. Create and activate a Python environment

Use Python 3.8 or newer. A virtual environment is recommended.

### 2. Install the Python packages

Install the dependencies listed above into the environment.

### 3. Install Wireshark / tshark

Make sure `tshark` runs from the command line before preprocessing packet captures.

## Workflow

### Step 1: Prepare the raw captures

If `NorFin_csv/` already contains CSV files, you can skip directly to training.

Place the `.pcapng` files into the expected folder structure under `NorFin_Lab/`.

The preprocessing script uses the folder name to derive the label. Folders follow the pattern:

- `AttackName (Label)`

Examples:

- `t50 (DoS)`
- `Hydra (Bruteforce)`
- `NMAP (Recon)`
- `Injections (Injections)`
- `benign (Benign)`

### Step 2: Preprocess packet captures into CSV

Run the preprocessing script from the repository root:

```powershell
python NorFin_Lab/scripts/preprocess_pcap.py --input_dir "C:\Users\Knut\Documents\Studie_D\Riku\NorFin_Lab" --output_dir "C:\Users\Knut\Documents\Studie_D\Riku\NorFin_csv" --window_size 10
```

What this does:

- Reads `.pcapng` files recursively from the input folder.
- Extracts packet-window features using non-overlapping windows.
- Writes one CSV per capture file into `NorFin_csv/`.
- Uses the folder name as the label source.

Useful options:

- `--window_size` controls the number of packets per window. Default: `10`
- `--redo_all` forces reprocessing and overwrites existing CSVs

### Step 3: Train the TensorFlow models

Run the training script:

```powershell
python NorFin_Lab/scripts/classify_tf.py
```

The training script will:

- Load all generated CSV files from `NorFin_csv/`
- Remove metadata columns such as `source_pcap`, `folder_type`, and `window_index`
- Drop the `ttl` feature before training
- Standardize features with `StandardScaler`
- Train multiple model architectures:
  - feedforward
  - cnn
  - lstm
  - attention
- Evaluate multiple sequence group sizes:
  - 1
  - 2
  - 3
  - 4
- Save trained artifacts and comparison outputs

### Step 4: Review outputs

Training creates outputs in `NorFin_Lab/models/`, including:

- `saved_models/` — timestamped trained model runs
- `tf_model.keras` — trained model file
- `scaler.joblib` — fitted feature scaler
- `label_encoder.joblib` — class label mapping
- `train.log` — training log
- confusion matrix images
- feature importance plots and CSV files
- comparison results in JSON format

## How the Data Pipeline Works

### Preprocessing

The preprocessing script converts packet captures into feature rows. Each row represents a packet window and includes traffic statistics such as:

- flow duration
- packet lengths and rates
- TCP flag counts
- protocol counts
- directional statistics
- other aggregated packet-level features

### Training

The training script reads the CSV windows and builds sequence-based samples. It supports grouped windows, meaning that multiple consecutive rows can be concatenated into a single training example.

Key conventions:

- `ttl` is always removed before training
- `source_pcap`, `folder_type`, and `window_index` are treated as metadata only
- labels are encoded automatically
- training/validation/test separation is handled by the script

## Reproducing the Project

To reproduce a full run from scratch:

1. Obtain the raw `.pcapng` data externally.
2. Place the files into the expected `NorFin_Lab/` subfolders.
3. Install Python dependencies and `tshark`.
4. Run preprocessing to generate CSV windows.
5. Run training to produce the final TensorFlow models and evaluation outputs.

If pre-generated CSV files are already available in `NorFin_csv/`, you can skip steps 1-4 and run training directly.

## Adding New Data

To add another traffic class or dataset subset:

1. Create a new folder using the same naming convention:
   - `DatasetName (Label)`
2. Place `.pcapng` files inside that folder.
3. Run preprocessing again.
4. Train the models on the regenerated CSV output.

## Troubleshooting

### `pyshark` cannot find `tshark`

Install Wireshark and ensure `tshark` is available on PATH.

### No pcap files are found

Verify that the input path points to the root of the raw capture directory and that the files are `.pcapng` or `.pcap`.

### Training cannot find CSV files

Make sure preprocessing has been completed and that `NorFin_csv/` contains generated CSV files in the expected folder structure.

### Memory or runtime issues

The preprocessing and training stages can be expensive on large capture sets. Use smaller subsets during testing if needed.

## Citation

If you use this repository or its outputs in your own work, please cite the associated paper:

**Securing IoT with AI-based Cyberattack Detection Models using Real-World Testbeds and Datasets**

## Acknowledgment
This research was supported by Business Finland (grants 2356/31/2023 and 8365/31/2022) and the University of Jyväskylä. In addition, it was funded by the European Union’s Horizon Research and Innovation Programme under Grant Agreement No. \#101120657 (Project ENFIELD: European Lighthouse to Manifest Trustworthy and Green AI), and by the Research Council of Norway through the SFI Norwegian Centre for Cybersecurity in Critical Sectors (NORCICS), Project No. \#310105, and through Strengthening Resilience in Critical Sectors through IT–OT Integration and Human–Organizational Aspects (ResCri) Project No. \#359829.