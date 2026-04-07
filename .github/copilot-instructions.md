# NorFin Network Intrusion Detection System

## Project Overview
This is a TensorFlow-based network traffic classifier for intrusion detection. The pipeline processes `.pcapng` files into windowed CSV features, then trains deep learning models to classify traffic as benign or various attack types (DoS, Bruteforce, Recon, Injections).

## Directory Structure
- `NorFin_Lab/` - Source pcapng files organized by attack type (e.g., `t50 (DoS)/`, `Hydra (Bruteforce)/`)
- `NorFin_csv/` - Preprocessed CSV windows, mirroring the folder structure
- `NorFin_Lab/scripts/` - All Python scripts
- `NorFin_Lab/models/` - Trained models, scalers, encoders, and comparison results

## Data Pipeline

### 1. Preprocessing (pcap → CSV)
```powershell
python preprocess_pcap.py --input_dir "path/to/NorFin_Lab" --output_dir "path/to/NorFin_csv" --window_size 10
```
- Extracts per-window features from non-overlapping packet windows (default: 10 packets)
- Features include: flow duration, rates, TCP flags, protocol counts, packet statistics
- Labels derived from folder names: `folder_type (label)` format (e.g., `t50 (DoS)`)

### 2. Training (CSV → Model)
Run `classify_tf.py` directly - `main()` handles everything:
```powershell
python classify_tf.py
```
- Trains multiple model architectures: `feedforward`, `cnn`, `lstm`, `attention`
- Tests multiple `group_sizes` (consecutive rows grouped as one sample): `[1, 2, 3, 4]`
- Automatic train/test/validation split by CSV file (not random rows)
- Sliding-window augmentation on training data only

## Key Conventions

### Feature Engineering
- The `ttl` column is **always dropped** to prevent model reliance on it
- Columns `source_pcap`, `folder_type`, `window_index` are metadata-only, dropped before training
- Features are standardized using `StandardScaler` (saved as `scaler.joblib`)

### Data Splitting Strategy
- Validation uses separate CSV files (25% of files per class, sorted alphabetically)
- Test windows are contiguous blocks within files (prevents temporal leakage)
- Training augmentation uses stride=1 sliding windows, excluding any overlap with test windows

### Model Artifacts (per run in `models/saved_models/model_TIMESTAMP/`)
- `tf_model.keras` - Trained model
- `scaler.joblib` - Feature scaler (fit on training data)
- `label_encoder.joblib` - Label mapping

### Logging & Comparisons
- All runs log to `models/train.log`
- Results JSON and plots saved to `models/comparisons/`

## Dependencies
```
tensorflow pandas scikit-learn joblib matplotlib seaborn pyshark
```
Note: `pyshark` requires `tshark` (Wireshark) installed and on PATH.

## Adding New Attack Types
1. Place `.pcapng` files in `NorFin_Lab/AttackName (Label)/`
2. Run preprocessing: `python preprocess_pcap.py ...`
3. CSVs auto-labeled from folder name pattern

## Quick Reference
| Task | Command/File |
|------|-------------|
| Preprocess pcaps | `preprocess_pcap.py --input_dir ... --output_dir ...` |
| Train all models | `python classify_tf.py` (runs `main()`) |
| Single prediction | `classify_tf.py --predict_csv single_row.csv --model_out path/to/model` |
| Feature importance | `feature_importances.py` (called from classify_tf) |
