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

# 1. Load Candidate Data safely
df_path = "csv/rs_training_data_real_candidates.csv"
if not os.path.exists(df_path):
    raise FileNotFoundError(f"Could not find {df_path}")

df = pd.read_csv(df_path)

# Convert non-numeric columns to numeric where possible
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_cols:
    converted = pd.to_numeric(df[col], errors='coerce')
    if not converted.isna().all():
        df[col] = converted

# 2. Load Neural Network Predictions
target_var = "hysteresis_window_V"
pred_file = f"csv/predictions_nn_{target_var}.csv"

if not os.path.exists(pred_file):
    raise FileNotFoundError(f"{pred_file} not found. Run nn.py first.")
pred_df = pd.read_csv(pred_file)
y_test = pd.to_numeric(pred_df["y_true"], errors='coerce').dropna().values
y_pred = pd.to_numeric(pred_df["y_pred"], errors='coerce').dropna().values

# 3. Load NN permutation importance (written by nn.py or nn_permutation_importance.py)
imp_file = f"csv/permutation_importance_nn_{target_var}.csv"
if not os.path.exists(imp_file):
    raise FileNotFoundError(f"{imp_file} not found. Run nn_permutation_importance.py (or nn.py) first.")
feat_imp = pd.read_csv(imp_file).head(8)

r2_val = r2_score(y_test, y_pred) if len(y_test) > 0 else 0.0
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred)) if len(y_test) > 0 else 0.0
residuals = y_test - y_pred


# ==========================================
# IMAGE 1: Neural Network Model Parity Plot
# ==========================================
fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
ax1.scatter(y_test, y_pred, color='#8B5CF6', edgecolors='#4C1D95', alpha=0.7, s=40, label='NN Test Candidates')
if len(y_test) > 0:
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax1.plot([min_v, max_v], [min_v, max_v], color='#EF4444', linestyle='--', linewidth=1.8, label='Ideal Parity (1:1)')
ax1.set_title(f'Neural Network Accuracy ($R^2 = {r2_val:.3f}$, RMSE = {rmse_val:.3f})', fontweight='bold', fontsize=11)
ax1.set_xlabel(f'Actual {target_var}', fontweight='bold', fontsize=10)
ax1.set_ylabel(f'Predicted {target_var}', fontweight='bold', fontsize=10)
ax1.legend(frameon=True, facecolor='white', loc='upper left')
ax1.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig1.savefig("plots/nn_plot_1_parity.png", dpi=300, bbox_inches='tight')
plt.close(fig1)


# ==========================================
# IMAGE 2: Residual Error Distribution (Neural Network)
# ==========================================
fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=300)
n, bins, _ = ax2.hist(residuals, bins=15, color='#A855F7', edgecolor='black', alpha=0.7, density=True)
if len(residuals) > 0:
    mu, std = norm.fit(residuals)
    x_axis = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x_axis, norm.pdf(x_axis, mu, std), color='#581C87', linewidth=2, label='Gaussian Fit')
ax2.axvline(0, color='black', linestyle='--', alpha=0.7)
ax2.set_title('NN Error Distribution (Residuals)', fontweight='bold', fontsize=11)
ax2.set_xlabel(r'Error ($y_{\mathrm{true}} - y_{\mathrm{pred}}$)', fontweight='bold', fontsize=10)
ax2.set_ylabel('Density', fontweight='bold', fontsize=10)
ax2.legend(frameon=True, facecolor='white')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig2.savefig("plots/nn_plot_2_residuals.png", dpi=300, bbox_inches='tight')
plt.close(fig2)


# ==========================================
# IMAGE 3: Candidate Screening Space
# ==========================================
fig3, ax3 = plt.subplots(figsize=(6, 5), dpi=300)
x_mismatch = "lattice_mismatch_pct" if "lattice_mismatch_pct" in df.columns else df.columns[0]
y_lumo = "dE_LUMO_eV" if "dE_LUMO_eV" in df.columns else df.columns[1]

clean_df = df[[x_mismatch, y_lumo]].dropna()
ax3.scatter(clean_df[x_mismatch], clean_df[y_lumo], color='#10B981', edgecolors='#064E3B', alpha=0.6, s=35)
ax3.set_title('Candidate Screening Space', fontweight='bold', fontsize=11)
ax3.set_xlabel(str(x_mismatch), fontweight='bold', fontsize=10)
ax3.set_ylabel(str(y_lumo), fontweight='bold', fontsize=10)
ax3.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig3.savefig("plots/nn_plot_3_screening.png", dpi=300, bbox_inches='tight')
plt.close(fig3)


# ==========================================
# IMAGE 4: Neural Network Feature Sensitivity / Weights
# ==========================================
fig4, ax4 = plt.subplots(figsize=(6, 5), dpi=300)
palette = sns.color_palette("Purples_r", n_colors=len(feat_imp))
ax4.barh(feat_imp['feature'][::-1], feat_imp['importance_mean'][::-1],
         xerr=feat_imp['importance_std'][::-1], color=palette, edgecolor='black', linewidth=0.5)
ax4.set_title(f'NN Permutation Importance: {target_var}', fontweight='bold', fontsize=11)
ax4.set_xlabel('Mean drop in test $R^2$ when feature is shuffled', fontweight='bold', fontsize=10)
ax4.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig4.savefig("plots/nn_plot_4_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close(fig4)

print("Generated 4 Neural Network plots:")
print(" - plots/nn_plot_1_parity.png")
print(" - plots/nn_plot_2_residuals.png")
print(" - plots/nn_plot_3_screening.png")
print(" - plots/nn_plot_4_feature_importance.png")