import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_squared_error

# Set global scientific plotting theme
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# Function to generate Parity and Residual plots for a given target metric
def generate_metric_plots(target_name, display_title, model_type="xgb"):
    pred_file = f"csv/predictions_{model_type}_{target_name}.csv"
    
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"{pred_file} not found. Run the {model_type} training script first.")
    pred_df = pd.read_csv(pred_file)
    y_test = pd.to_numeric(pred_df["y_true"], errors='coerce').dropna().values
    y_pred = pd.to_numeric(pred_df["y_pred"], errors='coerce').dropna().values

    r2_val = r2_score(y_test, y_pred) if len(y_test) > 0 else 0.0
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred)) if len(y_test) > 0 else 0.0
    residuals = y_test - y_pred

    # 1. Parity Plot
    fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
    ax1.scatter(y_test, y_pred, color='#3B82F6' if model_type == 'xgb' else '#8B5CF6', 
                edgecolors='#1E3A8A' if model_type == 'xgb' else '#4C1D95', alpha=0.7, s=40, label='Test Set')
    if len(y_test) > 0:
        min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax1.plot([min_v, max_v], [min_v, max_v], color='#EF4444', linestyle='--', linewidth=1.8, label='1:1 Parity')
    
    ax1.set_title(f'{display_title} Parity ($R^2 = {r2_val:.3f}$, RMSE = {rmse_val:.3f})', fontweight='bold', fontsize=10)
    ax1.set_xlabel(f'Actual {display_title}', fontweight='bold', fontsize=9)
    ax1.set_ylabel(f'Predicted {display_title}', fontweight='bold', fontsize=9)
    ax1.legend(frameon=True, facecolor='white', loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    fig1.savefig(f"plots/{model_type}_{target_name}_parity.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 2. Residual Distribution Plot
    fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=300)
    n, bins, _ = ax2.hist(residuals, bins=15, color='#F43F5E', edgecolor='black', alpha=0.7, density=True)
    if len(residuals) > 0:
        mu, std = norm.fit(residuals)
        x_axis = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x_axis, norm.pdf(x_axis, mu, std), color='#9F1239', linewidth=2, label='Gaussian Fit')
    
    ax2.axvline(0, color='black', linestyle='--', alpha=0.7)
    ax2.set_title(f'{display_title} Error Residuals', fontweight='bold', fontsize=10)
    ax2.set_xlabel(r'Error ($y_{\mathrm{true}} - y_{\mathrm{pred}}$)', fontweight='bold', fontsize=9)
    ax2.set_ylabel('Density', fontweight='bold', fontsize=9)
    ax2.legend(frameon=True, facecolor='white')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    fig2.savefig(f"plots/{model_type}_{target_name}_residuals.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

# Generate plots for On/Off Ratio
generate_metric_plots("log10_on_off_ratio", "Log10(On/Off Ratio)", model_type="xgb")

# Generate plots for Retention Time
generate_metric_plots("log10_retention_tau_s", "Log10(Retention Time τ [s])", model_type="xgb")

print("Generated plots for On/Off Ratio and Retention Time:")
print(" - plots/xgb_log10_on_off_ratio_parity.png")
print(" - plots/xgb_log10_on_off_ratio_residuals.png")
print(" - plots/xgb_log10_retention_tau_s_parity.png")
print(" - plots/xgb_log10_retention_tau_s_residuals.png")