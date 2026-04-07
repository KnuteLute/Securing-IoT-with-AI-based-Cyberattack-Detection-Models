"""
Feature importance utilities (permutation and SHAP) extracted from classify_tf.py
so that the main training script stays concise.

Provides:
 - compute_permutation_importances(model, X_test, y_test, scaler, ...)
 - compute_shap_importances(model, X_train_s, X_test_s, feature_names, ...)

The functions depend on numpy, pandas, matplotlib, seaborn and scikit-learn's accuracy_score.
SHAP is optional; compute_shap_importances will raise if shap is not installed.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
from sklearn.metrics import accuracy_score

try:
    shap = importlib.import_module('shap')
except Exception:
    shap = None


def compute_permutation_importances(model, X_test, y_test, scaler, n_repeats=5, random_state=42, outdir=None):
    """Compute permutation importance for each feature by measuring drop in accuracy.
    Returns a DataFrame with features and importance (higher = more important).
    """
    rng = np.random.RandomState(random_state)

    # Ensure we have DataFrame for column names
    if not hasattr(X_test, 'columns'):
        X_df = pd.DataFrame(X_test)
    else:
        X_df = X_test.copy()

    # baseline accuracy
    X_base = scaler.transform(X_df)
    probs_base = model.predict(X_base)
    y_pred_base = np.argmax(probs_base, axis=1)
    baseline_acc = accuracy_score(y_test, y_pred_base)

    records = []
    for col in X_df.columns:
        perm_accs = []
        for _ in range(n_repeats):
            X_perm = X_df.copy()
            arr = X_perm[col].to_numpy()
            X_perm[col] = rng.permutation(arr)
            Xp = scaler.transform(X_perm)
            probs = model.predict(Xp)
            y_pred = np.argmax(probs, axis=1)
            perm_accs.append(accuracy_score(y_test, y_pred))
        mean_perm = float(np.mean(perm_accs))
        imp = float(baseline_acc - mean_perm)
        records.append({'feature': col, 'importance': imp, 'perm_mean_acc': mean_perm})

    imp_df = pd.DataFrame(records).sort_values('importance', ascending=False).reset_index(drop=True)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        csvp = os.path.join(outdir, 'feature_importances_permutation.csv')
        imp_df.to_csv(csvp, index=False)

        plt.figure(figsize=(10, max(4, len(imp_df) * 0.25)))
        sns.barplot(x='importance', y='feature', data=imp_df, palette='viridis')
        plt.title('Permutation Feature Importances (accuracy drop)')
        plt.xlabel('Decrease in accuracy (baseline - permuted)')
        plt.tight_layout()
        pngp = os.path.join(outdir, 'feature_importances.png')
        plt.savefig(pngp)
        plt.close()

    return imp_df


def compute_shap_importances(model, X_train_s, X_test_s, feature_names, outdir=None, ns_background=50, ns_explain=50):
    """Compute SHAP importances using a small background sample and the provided model.
    Saves CSV and barplot and a SHAP summary plot (png) in outdir when provided.
    This function is optional and will raise if the shap package is not installed.
    """
    if shap is None:
        raise RuntimeError('shap package not installed. Install with: pip install shap')

    bsize = min(ns_background, len(X_train_s)) if len(X_train_s) > 0 else 0
    if bsize <= 0:
        raise ValueError('Not enough background samples for SHAP explanation')
    bg_idx = np.random.RandomState(42).choice(len(X_train_s), size=bsize, replace=False)
    background = X_train_s[bg_idx]

    n_explain = min(ns_explain, len(X_test_s))
    to_explain = X_test_s[:n_explain]

    explainer = shap.Explainer(model, background)
    logging.info(f'Computing SHAP values on {n_explain} samples (background size {bsize}) - this may take a while')
    shap_values = explainer(to_explain)

    vals = shap_values.values
    if vals is None:
        raise RuntimeError('SHAP returned no values')

    vals = np.array(vals)
    if vals.ndim == 3:
        mean_abs = np.mean(np.abs(vals), axis=(0, 2))
    elif vals.ndim == 2:
        mean_abs = np.mean(np.abs(vals), axis=0)
    else:
        raise RuntimeError(f'Unexpected shap values shape: {vals.shape}')

    imp_df = pd.DataFrame({'feature': list(feature_names), 'mean_abs_shap': mean_abs})
    imp_df = imp_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        csvp = os.path.join(outdir, 'feature_importances_shap.csv')
        imp_df.to_csv(csvp, index=False)

        plt.figure(figsize=(10, max(4, len(imp_df) * 0.25)))
        sns.barplot(x='mean_abs_shap', y='feature', data=imp_df, palette='magma')
        plt.title('SHAP mean(|value|) feature importances')
        plt.xlabel('Mean absolute SHAP value')
        plt.tight_layout()
        pngp = os.path.join(outdir, 'feature_importances_shap.png')
        plt.savefig(pngp)
        plt.close()

        try:
            fig = plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, to_explain, feature_names=feature_names, show=False)
            plt.tight_layout()
            splot = os.path.join(outdir, 'shap_summary.png')
            plt.savefig(splot)
            plt.close()
        except Exception as e:
            logging.warning(f'Failed to produce SHAP summary_plot: {e}')

    return imp_df
