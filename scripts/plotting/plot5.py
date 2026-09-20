import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Set global scientific plotting theme
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# 1. Load Dataset to identify feature column names
df_path = "csv/rs_training_data_real_candidates.csv"
if not os.path.exists(df_path):
    raise FileNotFoundError(f"{df_path} not found. Run Prepare_real_candidates.py first.")
df = pd.read_csv(df_path)
band_align_cols = [c for c in df.columns if c.startswith("band_align_")]
feature_cols = [c for c in [
    "dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
    "Nt_cm3", "eps_shell", "Vmax_V", "lattice_mismatch_pct",
] if c in df.columns] + band_align_cols

# Function to plot feature importance from XGBoost JSON model
def plot_xgb_feature_importance(target_name, display_title):
    model_file = f"xgb_model_{target_name}.json"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"{model_file} not found. Run kmeans_xgboost_train.py first.")

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(model_file)
    scores = xgb_model.feature_importances_
    feat_df = pd.DataFrame({'feature': feature_cols[:len(scores)], 'importance': scores})
    feat_df = feat_df.sort_values('importance', ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    palette = sns.color_palette("mako", n_colors=len(feat_df))
    ax.barh(feat_df['feature'][::-1], feat_df['importance'][::-1], color=palette[::-1], edgecolor='black', linewidth=0.5)
    
    ax.set_title(f'XGBoost Feature Importance: {display_title}', fontweight='bold', fontsize=10)
    ax.set_xlabel('Relative Importance Score', fontweight='bold', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(f"plots/xgb_{target_name}_importance.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# Function to plot Neural Network permutation importance
# (written by nn.py, or by nn_permutation_importance.py for previously saved models)
def plot_nn_feature_importance(target_name, display_title):
    imp_file = f"csv/permutation_importance_nn_{target_name}.csv"
    if not os.path.exists(imp_file):
        raise FileNotFoundError(f"{imp_file} not found. Run nn_permutation_importance.py (or nn.py) first.")
    feat_df = pd.read_csv(imp_file).head(8)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    palette = sns.color_palette("Purples_r", n_colors=len(feat_df))
    ax.barh(feat_df['feature'][::-1], feat_df['importance_mean'][::-1],
            xerr=feat_df['importance_std'][::-1], color=palette, edgecolor='black', linewidth=0.5)

    ax.set_title(f'NN Permutation Importance: {display_title}', fontweight='bold', fontsize=10)
    ax.set_xlabel('Mean drop in test $R^2$ when feature is shuffled', fontweight='bold', fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    fig.savefig(f"plots/nn_{target_name}_importance.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

targets = [("log10_on_off_ratio", "Log10(On/Off Ratio)"),
           ("log10_retention_tau_s", "Log10(Retention Time τ [s])")]

generated = []
for target_name, title in targets:
    plot_xgb_feature_importance(target_name, title)
    generated.append(f"plots/xgb_{target_name}_importance.png")

    if os.path.exists(f"csv/permutation_importance_nn_{target_name}.csv"):
        plot_nn_feature_importance(target_name, title)
        generated.append(f"plots/nn_{target_name}_importance.png")
    else:
        print(f"Skipping NN importance for {target_name}: no csv/permutation_importance_nn_{target_name}.csv "
              f"(model not trained yet).")

print("Generated Feature Importance Plots:")
for g in generated:
    print(f" - {g}")