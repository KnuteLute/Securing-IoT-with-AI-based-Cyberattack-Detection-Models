r"""
Train a TensorFlow classifier on the CSV windows produced by preprocessing and provide
an interface to classify a single row.

Usage examples (PowerShell):
  python classify_tf.py --data_dir "c:\Users\Knut\Documents\Studie_D\Riku\NorFin_csv" --model_out "c:\temp\nf_model" --epochs 20
  python classify_tf.py --predict_csv "c:\path\single_row.csv" --model_out "c:\temp\nf_model"

Dependencies:
  pip install tensorflow pandas scikit-learn joblib matplotlib seaborn

Behavior:
 - Recursively loads all .csv files under --data_dir
 - Drops columns: source_pcap, folder_type, window_index
 - Uses 'label' column as target (label-encoded)
 - Shuffles data, splits train/test
 - Standardizes numeric features (fit on train)
 - Trains a small dense neural network and saves model, scaler, encoder
 - Prints test accuracy and confusion matrix; saves confusion matrix image to model_out
 - If --predict_csv is provided, loads model and artifacts and predicts the label for that single-row CSV
"""

import os
import argparse
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau
import logging
import sys
from datetime import datetime
from feature_importances import compute_permutation_importances, compute_shap_importances



def prepare_data(df):
    # Remove unwanted columns
    drop_cols = [c for c in ['source_pcap', 'folder_type', 'window_index'] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Remove TTL from features to avoid model relying on it
    if 'ttl' in df.columns:
        df = df.drop(columns=['ttl'])

    # Ensure label present
    if 'label' not in df.columns:
        raise KeyError('label column not found in data')

    # Separate X and y (do not shuffle here; grouping should happen before shuffling)
    X = df.drop(columns=['label'])
    y = df['label'].astype(str)

    # If any non-numeric columns remain (shouldn't), try to coerce
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)

    return X, y


def group_consecutive_rows(X, y, group_size=1, stride=None):
    """Group consecutive rows into a single sample by concatenating features across time steps.

    - X: DataFrame of shape (N, F)
    - y: Series of length N
    - group_size: number of consecutive rows per new sample

    Returns (X_grouped, y_grouped). If N is not divisible by group_size, the remainder rows at the end are dropped.
    The new feature names are original_col__t0, original_col__t1, ..., preserving order.
    The label for each grouped sample is the mode (majority) of the group's labels; ties pick the first mode.
    """
    # If stride is None, use non-overlapping groups (stride == group_size)
    if stride is None:
        stride = group_size

    if group_size <= 1:
        starts = list(range(len(X)) )
        return X.reset_index(drop=True), y.reset_index(drop=True), starts

    n = len(X)
    # number of windows using given stride (start indices 0..n-group_size step=stride)
    if n < group_size:
        raise ValueError(f'Not enough rows ({n}) to form a single group of size {group_size}')
    starts = list(range(0, n - group_size + 1, stride))
    n_groups = len(starts)
    if n_groups == 0:
        raise ValueError(f'Not enough rows ({n}) to form a single group of size {group_size}')

    # Trim to full groups
    total = n_groups * group_size
    X_trim = X.iloc[:total].reset_index(drop=True)
    y_trim = y.iloc[:total].reset_index(drop=True)

    cols = list(X.columns)
    records = []
    labels = []
    for idx in starts:
        chunk = X.iloc[idx:idx+group_size]
        row = {}
        for t, (_, r) in enumerate(chunk.iterrows()):
            for col in cols:
                row[f'{col}__t{t}'] = r[col]
        # determine label by majority vote
        chunk_labels = y.iloc[idx:idx+group_size]
        mode = chunk_labels.mode()
        lbl = mode.iloc[0] if not mode.empty else chunk_labels.iloc[0]
        records.append(row)
        labels.append(lbl)

    X_group = pd.DataFrame.from_records(records)
    y_group = pd.Series(labels, name=y.name)
    # also return the starting indices of each grouped window to help prevent leakage
    return X_group, y_group, starts



def plot_confusion(y_true, y_pred, class_names, outpath):
    # y_true/y_pred are numeric-encoded labels; class_names provides tick labels
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def augment_sliding_windows(X_rows, y_rows, window_size, step=1, exclude_starts=None):
    """Generate sliding-window grouped samples from raw rows.

    - X_rows: DataFrame of shape (N, F) raw sequential rows
    - y_rows: Series of length N
    - window_size: number of consecutive rows per sample
    - step: stride between sliding windows (1 = fully overlapping)
    - exclude_starts: optional set of start indices to exclude (to avoid leakage into reserved test windows)

    Returns: (X_grouped, y_grouped)
    """
    if exclude_starts is None:
        exclude_starts = set()

    n = len(X_rows)
    starts = range(0, n - window_size + 1, step)
    records = []
    labels = []
    cols = list(X_rows.columns)
    for s in starts:
        if s in exclude_starts:
            continue
        chunk = X_rows.iloc[s:s+window_size]
        if len(chunk) < window_size:
            continue
        row = {}
        for t, (_, r) in enumerate(chunk.iterrows()):
            for col in cols:
                # For single-row windows, keep original column names to match non-augmented grouped windows
                if window_size == 1:
                    row[col] = r[col]
                else:
                    row[f'{col}__t{t}'] = r[col]
        chunk_labels = y_rows.iloc[s:s+window_size]
        mode = chunk_labels.mode()
        lbl = mode.iloc[0] if not mode.empty else chunk_labels.iloc[0]
        records.append(row)
        labels.append(lbl)

    X_group = pd.DataFrame.from_records(records)
    y_group = pd.Series(labels, name=y_rows.name)
    return X_group, y_group


def train_and_save(data_dir, model_out, epochs=20, test_size=0.2, batch_size=64, group_size=1, 
                   validation_ratio=0.25, model_type='feedforward', 
                   preprocessed_data=None):
    """
    Train and save model.
    
    Args:
        preprocessed_data: Optional dict with pre-grouped data to avoid re-grouping:
                    {'X_train_test': DataFrame, 'y_train_test': Series, 
                     'X_val': DataFrame, 'y_val': Series}
    """
    # Create timestamped directories for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create separate directories for models and images
    model_dir = os.path.join(model_out, 'saved_models', f'model_{timestamp}')
    image_dir = os.path.join(model_out, 'images', f'run_{timestamp}')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    logging.info(f'Model will be saved to: {model_dir}')
    logging.info(f'Images will be saved to: {image_dir}')
    
    # Use preprocessed data if available (expect raw rows cached), otherwise load and preprocess from scratch
    if preprocessed_data is not None and 'X_rows' in preprocessed_data:
        logging.info('Using cached preprocessed RAW rows (skipping load)')
        X_rows = preprocessed_data['X_rows']
        y_rows = preprocessed_data['y_rows']
        X_val_rows = preprocessed_data.get('X_val_rows')
        y_val_rows = preprocessed_data.get('y_val_rows')

        # Create non-overlapping grouped windows for splitting (stride=group_size)
        X_grouped, y_grouped, starts = group_consecutive_rows(X_rows, y_rows, group_size=group_size, stride=group_size)
        logging.info(f'After non-overlap grouping (group_size={group_size}): rows={len(X_grouped)}, cols={X_grouped.shape[1]}')
        X_val_grouped, y_val_grouped, val_starts = (None, None, None)
        if X_val_rows is not None and len(X_val_rows) > 0:
            X_val_grouped, y_val_grouped, val_starts = group_consecutive_rows(X_val_rows, y_val_rows, group_size=group_size, stride=group_size)
            logging.info(f'Validation set after non-overlap grouping: rows={len(X_val_grouped)}')
    else:
        logging.info('Loading and preprocessing data from scratch...')
        # Load data with automatic validation split
        df_train_test, df_validation = load_csvs_by_split(data_dir, validation_ratio=validation_ratio)
        # Prepare raw train/test rows
        # Keep source identifiers (if present) so we can pick contiguous test blocks per file
        src_rows = df_train_test['source_pcap'] if 'source_pcap' in df_train_test.columns else pd.Series(['_nosource_'] * len(df_train_test))
        X_rows, y_rows = prepare_data(df_train_test)
        logging.info(f'Train/test before grouping: rows={len(X_rows)}, cols={X_rows.shape[1]}')
        logging.info(f'Label distribution:\n{y_rows.value_counts(dropna=False).to_string()}')

        # Non-overlap grouping to use for train/test split
        X_grouped, y_grouped, starts = group_consecutive_rows(X_rows, y_rows, group_size=group_size, stride=group_size)
        logging.info(f'After non-overlap grouping (group_size={group_size}): rows={len(X_grouped)}, cols={X_grouped.shape[1]}')

        # Prepare validation raw rows and non-overlap grouping if available
        X_val_rows, y_val_rows = None, None
        X_val_grouped, y_val_grouped, val_starts = None, None, None
        if df_validation is not None and len(df_validation) > 0:
            X_val_rows, y_val_rows = prepare_data(df_validation)
            X_val_grouped, y_val_grouped, val_starts = group_consecutive_rows(X_val_rows, y_val_rows, group_size=group_size, stride=group_size)
            logging.info(f'Validation set after non-overlap grouping: rows={len(X_val_grouped)}')

    # Now, use the non-overlap grouped windows (X_grouped, y_grouped) for splitting into train/test
    # Fit label encoder on grouped labels so we can stratify
    le = LabelEncoder()
    y_group_enc = le.fit_transform(y_grouped)

    # Instead of random window-level split, pick contiguous blocks per source to make excluded
    # starts smaller and avoid sprinkling test windows across the file.
    # We need the source mapping (src_rows) — if absent, treat entire X_rows as single source.
    src_series = preprocessed_data.get('src_rows') if preprocessed_data is not None else None
    if src_series is None:
        src_series = pd.Series(['_nosource_'] * len(X_rows))

    # Build mapping from grouped window index -> source and window start
    # starts is the list of raw-row start indices used to build X_grouped
    grouped_source = []
    for s in starts:
        grouped_source.append(src_series.iloc[s])

    # For each source, collect indices of grouped windows and choose a contiguous block for test
    test_idx = []
    train_idx = []
    rng = np.random.RandomState(42)
    unique_sources = pd.Series(grouped_source).unique()
    for src in unique_sources:
        idxs = [i for i, v in enumerate(grouped_source) if v == src]
        if not idxs:
            continue
        n = len(idxs)
        n_test = max(1, int(n * test_size)) if n > 1 else 0
        if n_test == 0:
            train_idx.extend(idxs)
            continue
        # Select a contiguous block of length n_test within idxs
        start_pos = rng.randint(0, max(1, n - n_test + 1))
        block = idxs[start_pos:start_pos + n_test]
        test_idx.extend(block)
        # remaining go to train
        train_idx.extend([i for i in idxs if i not in block])

    # Ensure indices are numpy arrays
    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))

    X_train_grouped = X_grouped.iloc[train_idx].reset_index(drop=True)
    y_train_grouped = y_grouped.iloc[train_idx].reset_index(drop=True)
    X_test_grouped = X_grouped.iloc[test_idx].reset_index(drop=True)
    y_test_grouped = y_grouped.iloc[test_idx].reset_index(drop=True)

    # Determine which raw rows belong to test windows (for exclusion when augmenting)
    test_starts = [starts[i] for i in test_idx]
    test_row_set = set()
    for s in test_starts:
        test_row_set.update(range(s, s + group_size))

    # Build exclude_starts: any sliding-window start whose window would overlap test_row_set
    n_rows = len(X_rows)
    possible_starts = range(0, n_rows - group_size + 1)
    exclude_starts = set()
    for s in possible_starts:
        if any(i in test_row_set for i in range(s, s + group_size)):
            exclude_starts.add(s)

    # Augment training set using sliding windows (stride=1) excluding windows overlapping test windows
    X_aug, y_aug = augment_sliding_windows(X_rows, y_rows, window_size=group_size, step=1, exclude_starts=exclude_starts)
    logging.info(f'Augmented training windows: {len(X_aug)} (excluded {len(exclude_starts)} starts overlapping test windows)')

    # Combine non-overlap train windows with augmented windows and deduplicate
    if len(X_train_grouped) > 0 and len(X_aug) > 0:
        tmp_train = pd.concat([
            X_train_grouped.assign(__label=y_train_grouped.values),
            X_aug.assign(__label=y_aug.values)
        ], ignore_index=True)
    elif len(X_train_grouped) > 0:
        tmp_train = X_train_grouped.assign(__label=y_train_grouped.values)
    else:
        tmp_train = X_aug.assign(__label=y_aug.values)

    feature_cols = [c for c in tmp_train.columns if c != '__label']
    before = len(tmp_train)
    tmp_train = tmp_train.drop_duplicates(subset=feature_cols, keep='first').reset_index(drop=True)
    after = len(tmp_train)
    logging.info(f'Combined train windows deduplicated: before={before}, after={after}')

    # Final training arrays
    X_train_final = tmp_train[feature_cols]
    y_train_final = le.transform(tmp_train['__label'])

    # Final test arrays (do not augment)
    X_test_final = X_test_grouped
    y_test_final = le.transform(y_test_grouped)

    # Fit scaler on training features and transform test/val
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_final)
    # Ensure test frame has same columns/order as training features
    if isinstance(X_test_final, pd.DataFrame):
        X_test_final = X_test_final.reindex(columns=feature_cols, fill_value=0.0)
    X_test_s = scaler.transform(X_test_final)

    # Prepare validation data if available (use non-overlap grouped validation)
    X_val_s, y_val_enc = None, None
    if 'X_val_grouped' in locals() and X_val_grouped is not None and len(X_val_grouped) > 0:
        # Only transform labels if they exist
        if 'y_val_grouped' in locals() and y_val_grouped is not None and len(y_val_grouped) > 0:
            y_val_enc = le.transform(y_val_grouped)
        else:
            y_val_enc = None

        # Transform features (reindex to match training features)
        if X_val_grouped is not None:
            if isinstance(X_val_grouped, pd.DataFrame):
                X_val_grouped = X_val_grouped.reindex(columns=feature_cols, fill_value=0.0)
            X_val_s = scaler.transform(X_val_grouped)

        logging.info(f'Validation set ready: {X_val_s.shape if X_val_s is not None else "None"}')

    num_classes = len(np.unique(np.concatenate([y_train_final, y_test_final])))
    model = build_model(X_train_s.shape[1], num_classes, group_size=group_size, model_type=model_type)

    logging.info(f"Model architecture: {model_type}")
    logging.info(f"Total parameters: {model.count_params():,}")

    logging.info(f"\nStarting training: epochs={epochs}, train_shape={X_train_s.shape}, test_shape={X_test_s.shape}")
    
    # Use validation_data during training if available
    validation_data = (X_val_s, y_val_enc) if (X_val_s is not None and y_val_enc is not None) else None
    if validation_data is not None:
        logging.info(f"Using validation set during training: {X_val_s.shape if X_val_s is not None else 'None'}")
    
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    # Callback to compute macro-F1 on train and validation sets after each epoch
    class F1Callback(tf.keras.callbacks.Callback):
        def __init__(self, X_train, y_train, X_val=None, y_val=None, pred_batch_size=1024):
            super().__init__()
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.pred_batch_size = pred_batch_size
            self.train_f1 = []
            self.val_f1 = []

        def on_epoch_end(self, epoch, logs=None):
            try:
                y_train_probs = self.model.predict(self.X_train, batch_size=self.pred_batch_size, verbose=0)
                y_train_pred = np.argmax(y_train_probs, axis=1)
                f1t = f1_score(self.y_train, y_train_pred, average='macro')
            except Exception as e:
                logging.warning(f'Failed to compute train F1 at epoch {epoch}: {e}')
                f1t = None
            self.train_f1.append(float(f1t) if f1t is not None else None)

            if self.X_val is not None and self.y_val is not None:
                try:
                    y_val_probs = self.model.predict(self.X_val, batch_size=self.pred_batch_size, verbose=0)
                    y_val_pred = np.argmax(y_val_probs, axis=1)
                    f1v = f1_score(self.y_val, y_val_pred, average='macro')
                except Exception as e:
                    logging.warning(f'Failed to compute val F1 at epoch {epoch}: {e}')
                    f1v = None
            else:
                f1v = None
            self.val_f1.append(float(f1v) if f1v is not None else None)

            if logs is not None:
                logs['train_f1'] = self.train_f1[-1]
                logs['val_f1'] = self.val_f1[-1]

    f1_cb = F1Callback(X_train_s, y_train_final, X_val_s, y_val_enc, pred_batch_size=batch_size)
    callbacks = [lr_scheduler, f1_cb]
    history = model.fit(X_train_s, y_train_final, validation_data=validation_data, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callbacks)

    # Evaluate on test set (same CSV sources as training)
    test_loss, test_acc = model.evaluate(X_test_s, y_test_final, verbose=0)
    logging.info(f'\n=== Test Set (same CSVs as training) ===')
    logging.info(f'Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}')

    # Predictions on test set
    y_pred_probs = model.predict(X_test_s)
    y_pred = np.argmax(y_pred_probs, axis=1)

    labels = list(le.classes_)
    plot_confusion(y_test_final, y_pred, labels, outpath=os.path.join(image_dir, 'confusion_matrix_test.png'))
    
    # Compute macro F1 for test
    test_f1_macro = f1_score(y_test_final, y_pred, average='macro')
    logging.info('Test classification report:\n%s', classification_report(y_test_final, y_pred, target_names=labels))
    logging.info(f'Test macro F1: {test_f1_macro:.4f}')

    # Evaluate on validation set (unseen CSVs) only if both features and labels are available
    val_loss, val_acc, val_f1_macro = None, None, None
    if X_val_s is not None and y_val_enc is not None:
        val_loss, val_acc = model.evaluate(X_val_s, y_val_enc, verbose=0)
        logging.info(f'\n=== Validation Set (unseen CSVs) ===')
        logging.info(f'Validation accuracy: {val_acc:.4f}, loss: {val_loss:.4f}')

        # Predictions on validation
        y_val_pred_probs = model.predict(X_val_s)
        y_val_pred = np.argmax(y_val_pred_probs, axis=1)

        # Compute macro F1 for validation
        val_f1_macro = f1_score(y_val_enc, y_val_pred, average='macro')

        plot_confusion(y_val_enc, y_val_pred, labels, outpath=os.path.join(image_dir, 'confusion_matrix_validation.png'))
        logging.info('Validation classification report:\n%s', classification_report(y_val_enc, y_val_pred, target_names=labels))
        logging.info(f'Validation macro F1: {val_f1_macro:.4f}')
    elif X_val_s is not None and y_val_enc is None:
        logging.info('Validation features available but labels missing; skipping validation evaluation.')
    
    # compute and save permutation feature importances
    try:
        print("skipping permutation importances")
        #imp_df = compute_permutation_importances(model, X_test, y_test, scaler, n_repeats=5, random_state=42, outdir=image_dir)
        #logging.info('Top features by permutation importance:\n%s', imp_df.head(10).to_string(index=False))
    except Exception as e:
        logging.warning(f'Failed to compute permutation importances: {e}')

    # compute SHAP importances if shap is installed (the function will raise if shap missing)
    try:
        print("skipping SHAP importances")
        feat_names = X_train_final.columns if hasattr(X_train_final, 'columns') else [f'f{i}' for i in range(X_train_s.shape[1])]
        #shap_imp = compute_shap_importances(model, X_train_s, X_test_s, feat_names, outdir=image_dir, ns_background=100, ns_explain=200)
        #logging.info('Top features by SHAP mean(|value|):\n%s', shap_imp.head(10).to_string(index=False))
    except RuntimeError as re:
        logging.info('shap not available; skipping SHAP importance. To enable, pip install shap')
    except Exception as e:
        logging.warning(f'Failed to compute SHAP importances: {e}')

    # Save model and artifacts
    keras_path = os.path.join(model_dir, 'tf_model.keras')
    logging.info(f'\nSaving model to {keras_path}')
    model.save(keras_path)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.joblib'))
    logging.info(f'Model and artifacts saved to {model_dir}')
    logging.info(f'Images saved to {image_dir}')
    
    # Compute train accuracy/loss and macro F1
    train_loss, train_acc = model.evaluate(X_train_s, y_train_final, verbose=0)
    y_train_pred = np.argmax(model.predict(X_train_s), axis=1)
    train_f1_macro = f1_score(y_train_final, y_train_pred, average='macro')
    
    logging.info(f'\n=== Train Set ===')
    logging.info(f'Train accuracy: {train_acc:.4f}, loss: {train_loss:.4f}, macro F1: {train_f1_macro:.4f}')
    
    # Collect results including history for plotting
    results = {
        'group_size': group_size,
        'model_type': model_type,
        'epochs': epochs,
        'history': {
            'train_acc': history.history['accuracy'],
            'train_loss': history.history['loss'],
            'val_acc': history.history.get('val_accuracy', []),
            'val_loss': history.history.get('val_loss', []),
            # include per-epoch macro F1 (train and validation) collected by the F1 callback
            'train_f1': f1_cb.train_f1 if 'f1_cb' in locals() else [],
            'val_f1': f1_cb.val_f1 if 'f1_cb' in locals() else []
        },
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'test_f1_macro': float(test_f1_macro),
        'train_accuracy': float(train_acc),
        'train_loss': float(train_loss),
        'train_f1_macro': float(train_f1_macro),
        'val_accuracy': float(val_acc) if val_acc is not None else None,
        'val_loss': float(val_loss) if val_loss is not None else None,
        'val_f1_macro': float(val_f1_macro) if val_f1_macro is not None else None,
        'model_dir': model_dir,
        'image_dir': image_dir,
        'timestamp': timestamp
    }
    
    return results


def predict_single_row(predict_csv, model_out):
    # load artifacts
    scaler = joblib.load(os.path.join(model_out, 'scaler.joblib'))
    le = joblib.load(os.path.join(model_out, 'label_encoder.joblib'))
    
    # Load the .keras model
    keras_path = os.path.join(model_out, 'tf_model.keras')
    if os.path.exists(keras_path):
        model = models.load_model(keras_path)
    else:
        raise FileNotFoundError(f'No model found at {keras_path}')

    df = pd.read_csv(predict_csv)
    # Drop unwanted columns
    # also drop 'ttl' so prediction uses the same features as training
    drop_cols = [c for c in ['source_pcap', 'folder_type', 'window_index', 'ttl'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    X = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    # transform using DataFrame to avoid 'X does not have valid feature names' warning
    X_s = scaler.transform(X)
    probs = model.predict(X_s)
    preds = np.argmax(probs, axis=1)
    labels = le.inverse_transform(preds)
    for i,(lab,prob) in enumerate(zip(labels, probs)):
        logging.info(f'Row {i}: predicted={lab}, probs={prob}')
    return labels, probs


def load_csvs_by_split(root_dir, validation_ratio=0.25):
    """Load CSVs and automatically split into train/test vs validation sets per class.
    
    For each class (based on folder name), reserves validation_ratio of CSV files for validation.
    
    Args:
        root_dir: Root directory containing CSV files
        validation_ratio: Fraction of CSV files per class to use for validation (default 0.25 = 25%)
    
    Returns:
        train_test_df: DataFrame for train/test split
        validation_df: DataFrame for validation
    """
    paths = glob.glob(os.path.join(root_dir, '**', '*.csv'), recursive=True)
    logging.info(f"Found {len(paths)} CSV files under {root_dir}")
    
    if not paths:
        raise FileNotFoundError(f'No CSV files found under {root_dir}')
    
    # Group paths by parent folder (class)
    from collections import defaultdict
    class_files = defaultdict(list)
    
    for p in paths:
        # Extract folder name as class identifier
        parent_folder = os.path.basename(os.path.dirname(p))
        class_files[parent_folder].append(p)
    
    logging.info(f"Found {len(class_files)} classes:")
    for cls, files in class_files.items():
        logging.info(f"  {cls}: {len(files)} CSV files")
    
    train_test_paths = []
    validation_paths = []
    
    # For each class, split files into train/test and validation
    for cls, files in class_files.items():
        n_files = len(files)
        n_val = max(1, int(n_files * validation_ratio))  # At least 1 file for validation
        n_train_test = n_files - n_val
        
        if n_train_test < 1:
            logging.warning(f"Class {cls} has only {n_files} file(s). Using all for train/test, none for validation.")
            train_test_paths.extend(files)
        else:
            # Sort for reproducibility, then split
            files_sorted = sorted(files)
            train_test_paths.extend(files_sorted[:n_train_test])
            validation_paths.extend(files_sorted[n_train_test:])
            logging.info(f"  {cls}: {n_train_test} files for train/test, {n_val} files for validation")
    
    logging.info(f"\nTotal: {len(train_test_paths)} CSVs for train/test, {len(validation_paths)} CSVs for validation")
    
    # Load train/test CSVs
    train_test_dfs = []
    for p in train_test_paths:
        try:
            df = pd.read_csv(p)
            logging.info(f"Train/test: {os.path.basename(p)}: rows={len(df)}")
            train_test_dfs.append(df)
        except Exception as e:
            logging.warning(f'Failed to read {p}: {e}')
    
    # Load validation CSVs
    validation_dfs = []
    for p in validation_paths:
        try:
            df = pd.read_csv(p)
            logging.info(f"Validation: {os.path.basename(p)}: rows={len(df)}")
            validation_dfs.append(df)
        except Exception as e:
            logging.warning(f'Failed to read {p}: {e}')
    
    train_test_df = pd.concat(train_test_dfs, ignore_index=True) if train_test_dfs else pd.DataFrame()
    validation_df = pd.concat(validation_dfs, ignore_index=True) if validation_dfs else None
    
    logging.info(f"\nFinal shapes:")
    logging.info(f"  Train/test: {train_test_df.shape}")
    if validation_df is not None:
        logging.info(f"  Validation: {validation_df.shape}")
    
    return train_test_df, validation_df


def plot_comparison(results_list, output_dir):
    """Plot learning curves showing accuracy/loss over epochs for different group sizes.
    
    Creates separate plots for train, test, and validation sets showing how 
    accuracy changes across epochs for each group size.
    
    Args:
        results_list: List of result dictionaries from train_and_save
        output_dir: Directory to save plots
    """
    import json
    
    # Sort by group_size
    results_list = sorted(results_list, key=lambda x: x['group_size'])
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Save results to JSON (without numpy arrays in history)
    results_for_json = []
    for r in results_list:
        r_copy = r.copy()
        # Convert history lists to plain Python lists for JSON serialization
        if 'history' in r_copy:
            r_copy['history'] = {k: [float(v) for v in vals] for k, vals in r_copy['history'].items()}
        results_for_json.append(r_copy)
    
    json_path = os.path.join(comparison_dir, f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(json_path, 'w') as f:
        json.dump(results_for_json, f, indent=2)
    logging.info(f'Results saved to {json_path}')
    
    # Define colors for each group size
    colors = cm.get_cmap('viridis')(np.linspace(0, 0.9, len(results_list)))
    
    # Plot 1: Train Accuracy over Epochs
    plt.figure(figsize=(12, 7))
    for i, r in enumerate(results_list):
        epochs_range = range(1, len(r['history']['train_acc']) + 1)
        plt.plot(epochs_range, r['history']['train_acc'], 
                label=f"Group Size {r['group_size']}", 
                linewidth=2.5, marker='o', markersize=4, color=colors[i])
    
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Accuracy', fontsize=13)
    plt.title('Training Accuracy over Epochs', fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    train_acc_plot = os.path.join(comparison_dir, f'learning_curve_train_acc_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(train_acc_plot, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f'Train accuracy learning curve saved to {train_acc_plot}')
    
    # Plot 2: Validation Accuracy over Epochs (if available)
    has_val = any(r['history']['val_acc'] for r in results_list)
    if has_val:
        plt.figure(figsize=(12, 7))
        for i, r in enumerate(results_list):
            if r['history']['val_acc']:
                epochs_range = range(1, len(r['history']['val_acc']) + 1)
                plt.plot(epochs_range, r['history']['val_acc'], 
                        label=f"Group Size {r['group_size']}", 
                        linewidth=2.5, marker='s', markersize=4, color=colors[i])
        
        plt.xlabel('Epoch', fontsize=13)
        plt.ylabel('Accuracy', fontsize=13)
        plt.title('Validation Accuracy over Epochs', fontsize=15, fontweight='bold', pad=15)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        val_acc_plot = os.path.join(comparison_dir, f'learning_curve_val_acc_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(val_acc_plot, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f'Validation accuracy learning curve saved to {val_acc_plot}')
    
    # Plot 3: Combined Train + Val Accuracy in subplots
    if has_val:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        
        # Train accuracy
        for i, r in enumerate(results_list):
            epochs_range = range(1, len(r['history']['train_acc']) + 1)
            ax1.plot(epochs_range, r['history']['train_acc'], 
                    label=f"Group Size {r['group_size']}", 
                    linewidth=2.5, marker='o', markersize=4, color=colors[i])
        ax1.set_xlabel('Epoch', fontsize=13)
        ax1.set_ylabel('Accuracy', fontsize=13)
        ax1.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Validation accuracy
        for i, r in enumerate(results_list):
            if r['history']['val_acc']:
                epochs_range = range(1, len(r['history']['val_acc']) + 1)
                ax2.plot(epochs_range, r['history']['val_acc'], 
                        label=f"Group Size {r['group_size']}", 
                        linewidth=2.5, marker='s', markersize=4, color=colors[i])
        ax2.set_xlabel('Epoch', fontsize=13)
        ax2.set_ylabel('Accuracy', fontsize=13)
        ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle('Learning Curves: Accuracy over Epochs', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        combined_plot = os.path.join(comparison_dir, f'learning_curve_combined_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(combined_plot, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f'Combined learning curve saved to {combined_plot}')
    
    # Plot 4: Final Metrics Comparison (bar chart)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    group_sizes = [r['group_size'] for r in results_list]
    train_acc = [r['train_accuracy'] for r in results_list]
    test_acc = [r['test_accuracy'] for r in results_list]
    val_acc_final = [r['val_accuracy'] if r['val_accuracy'] is not None else 0 for r in results_list]
    
    train_f1 = [r.get('train_f1_macro', 0) for r in results_list]
    test_f1 = [r.get('test_f1_macro', 0) for r in results_list]
    val_f1 = [r.get('val_f1_macro', 0) if r.get('val_f1_macro') is not None else 0 for r in results_list]
    
    x = np.arange(len(group_sizes))
    width = 0.25
    
    # Accuracy comparison
    axes[0, 0].bar(x - width, train_acc, width, label='Train', color='#2ecc71')
    axes[0, 0].bar(x, test_acc, width, label='Test', color='#3498db')
    axes[0, 0].bar(x + width, val_acc_final, width, label='Validation', color='#e74c3c')
    axes[0, 0].set_xlabel('Group Size', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Final Accuracy Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(group_sizes)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Macro F1 comparison
    axes[0, 1].bar(x - width, train_f1, width, label='Train', color='#2ecc71')
    axes[0, 1].bar(x, test_f1, width, label='Test', color='#3498db')
    axes[0, 1].bar(x + width, val_f1, width, label='Validation', color='#e74c3c')
    axes[0, 1].set_xlabel('Group Size', fontsize=12)
    axes[0, 1].set_ylabel('Macro F1 Score', fontsize=12)
    axes[0, 1].set_title('Final Macro F1 Score Comparison', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(group_sizes)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Train Loss over epochs
    for i, r in enumerate(results_list):
        epochs_range = range(1, len(r['history']['train_loss']) + 1)
        axes[1, 0].plot(epochs_range, r['history']['train_loss'], 
                       label=f"Group Size {r['group_size']}", 
                       linewidth=2, color=colors[i])
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('Training Loss over Epochs', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Val Loss over epochs
    if has_val:
        for i, r in enumerate(results_list):
            if r['history']['val_loss']:
                epochs_range = range(1, len(r['history']['val_loss']) + 1)
                axes[1, 1].plot(epochs_range, r['history']['val_loss'], 
                               label=f"Group Size {r['group_size']}", 
                               linewidth=2, color=colors[i])
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Loss', fontsize=12)
        axes[1, 1].set_title('Validation Loss over Epochs', fontsize=13, fontweight='bold')
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    comprehensive_plot = os.path.join(comparison_dir, f'comprehensive_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(comprehensive_plot, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f'Comprehensive comparison saved to {comprehensive_plot}')
    
    # Print summary table
    logging.info('\n' + '='*100)
    logging.info('SUMMARY TABLE')
    logging.info('='*100)
    logging.info(f'{"Group":<8} {"Train Acc":<12} {"Test Acc":<12} {"Val Acc":<12} {"Train F1":<12} {"Test F1":<12} {"Val F1":<12}')
    logging.info('-'*100)
    for r in results_list:
        val_acc_str = f"{r['val_accuracy']:.4f}" if r['val_accuracy'] is not None else "N/A"
        val_f1_str = f"{r.get('val_f1_macro', 0):.4f}" if r.get('val_f1_macro') is not None else "N/A"
        logging.info(f"{r['group_size']:<8} {r['train_accuracy']:<12.4f} {r['test_accuracy']:<12.4f} {val_acc_str:<12} "
                    f"{r.get('train_f1_macro', 0):<12.4f} {r.get('test_f1_macro', 0):<12.4f} {val_f1_str:<12}")
    logging.info('='*100)
    logging.info(f"All comparison plots saved to {comparison_dir}\n")


def build_model(input_dim, num_classes, group_size=1, model_type='feedforward'):
    """
    Build different model architectures.
    
    Args:
        input_dim: Total number of features
        num_classes: Number of output classes
        group_size: Number of packets grouped together
        model_type: 'feedforward', 'cnn', 'lstm', or 'attention'
    """
    if model_type == 'feedforward':
        # Your current simple model
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    elif model_type == 'cnn':
        features_per_packet = input_dim // group_size
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Reshape((group_size, features_per_packet)),
            layers.Conv1D(128, kernel_size=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Conv1D(64, kernel_size=2, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    elif model_type == 'lstm':
        features_per_packet = input_dim // group_size
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Reshape((group_size, features_per_packet)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    elif model_type == 'attention':
        features_per_packet = input_dim // group_size
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Reshape((group_size, features_per_packet))(inputs)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        attn = layers.Dropout(0.2)(attn)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'NorFin_csv')
    model_out = os.path.join(os.path.dirname(__file__), '..', 'models')
    epochs = 30  # Increase epochs for deeper models
    group_sizes = [1, 2, 3, 4]  # Test different sequence lengths
    model_types = ['feedforward', 'cnn', 'lstm', 'attention']
    validation_ratio = 0.20
    
    # Setup logging once before loop
    os.makedirs(model_out, exist_ok=True)
    log_file = os.path.join(model_out, 'train.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    console.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(console)

    logging.info(f"\n{'='*80}")
    logging.info(f"Starting multi-group training run")
    logging.info(f"{'='*80}")
    logging.info(f"  data_dir={data_dir}")
    logging.info(f"  model_out={model_out}")
    logging.info(f"  epochs={epochs}")
    logging.info(f"  group_sizes={group_sizes}")
    logging.info(f"  validation_ratio={validation_ratio}")
    logging.info(f"{'='*80}\n")
    
    # Collect results from all runs
    all_results = []
    
    # OPTIMIZED: Pre-load CSV data ONCE, then preprocess once per group_size
    logging.info(f"\n{'*'*80}")
    logging.info("PRE-LOADING CSV DATA (done once for all experiments)")
    logging.info(f"{'*'*80}\n")
    
    df_train_test, df_validation = None, None
    try:
        df_train_test, df_validation = load_csvs_by_split(data_dir, validation_ratio=validation_ratio)
        logging.info(f"✓ CSV data loaded: train/test={df_train_test.shape}, validation={df_validation.shape if df_validation is not None else 'None'}\n")
    except Exception:
        logging.exception("Failed to pre-load CSV data.")
        return
    
    # Loop by group_size (preprocess once per group_size, reuse for all models)
    for group_size in group_sizes:
        logging.info(f"\n{'='*80}")
        logging.info(f"║ PROCESSING GROUP SIZE: {group_size}")
        logging.info(f"║ Preprocessing data once, then training {len(model_types)} models")
        logging.info(f"{'='*80}\n")
        
        # Preprocess and group data ONCE for this group_size
        logging.info(f"Preprocessing train/test data for group_size={group_size}...")
        # Prepare raw rows (no grouping here) and cache them for train_and_save which will
        # perform non-overlapping grouping for splitting and sliding-window augmentation only on train.
        X_rows, y_rows = prepare_data(df_train_test)
        logging.info(f'  Raw rows available: {len(X_rows)} rows, {X_rows.shape[1]} features')

        # Ensure we have a source identifier per raw row for per-file contiguous splitting
        if 'source_pcap' in df_train_test.columns:
            src_rows = df_train_test['source_pcap']
        else:
            src_rows = pd.Series(['_nosource_'] * len(df_train_test))

        X_val_rows, y_val_rows = None, None
        src_val_rows = None
        if df_validation is not None and len(df_validation) > 0:
            logging.info(f"Preprocessing validation raw rows for group_size={group_size}...")
            src_val_rows = df_validation['source_pcap'] if 'source_pcap' in df_validation.columns else pd.Series(['_nosource_'] * len(df_validation))
            X_val_rows, y_val_rows = prepare_data(df_validation)
            logging.info(f'  Validation raw rows: {len(X_val_rows)} rows')

        # Cache preprocessed raw rows for training
        preprocessed_data = {
            'X_rows': X_rows,
            'y_rows': y_rows,
            'src_rows': src_rows,
            'X_val_rows': X_val_rows,
            'y_val_rows': y_val_rows,
            'src_val_rows': src_val_rows
        }
        
        logging.info(f"✓ Data preprocessing complete for group_size={group_size}")
        logging.info(f"  Training {len(model_types)} models using this preprocessed data...\n")
        
        # Train all model types on this preprocessed data
        for model_type in model_types:
            logging.info(f"\n{'#'*80}")
            logging.info(f"Training: group_size={group_size}, model_type={model_type}")
            logging.info(f"{'#'*80}\n")
            
            try:
                results = train_and_save(
                    data_dir, model_out, 
                    epochs=epochs, 
                    group_size=group_size,
                    model_type=model_type,
                    validation_ratio=validation_ratio,
                    preprocessed_data=preprocessed_data  # Pass preprocessed data
                )
                all_results.append(results)
                logging.info(f"\n✓ Completed: group_size={group_size}, model_type={model_type}")
            except Exception:
                logging.exception(f'Failed: group_size={group_size}, model_type={model_type}')
        
        logging.info(f"\n{'='*80}")
        logging.info(f"║ COMPLETED ALL MODELS FOR GROUP SIZE {group_size}")
        logging.info(f"{'='*80}\n")
    
    # Plot comparison after all runs complete
    if all_results:
        logging.info(f"\n{'='*80}")
        logging.info(f"All training runs completed. Generating comparison plots...")
        logging.info(f"{'='*80}\n")
        try:
            plot_comparison(all_results, model_out)
        except Exception:
            logging.exception('Failed to generate comparison plots')
    else:
        logging.warning('No results to plot!')


if __name__ == '__main__':
    # Just press start
    main()
