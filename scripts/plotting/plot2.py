import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

# Set global scientific plotting theme
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# 1. Load Data safely
df_path = "csv/rs_training_data_real_candidates.csv"
if not os.path.exists(df_path):
    raise FileNotFoundError(f"Could not find {df_path}")

df = pd.read_csv(df_path)

# Convert non-numeric columns to numeric where possible (Pandas 3.x compatible)
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
for col in non_numeric_cols:
    converted = pd.to_numeric(df[col], errors='coerce')
    if not converted.isna().all():
        df[col] = converted

# 2. Load predictions file
target_var = "hysteresis_window_V"
pred_file = f"csv/predictions_xgb_{target_var}.csv"
if not os.path.exists(pred_file):
    raise FileNotFoundError(f"{pred_file} not found. Run kmeans_xgboost_train.py first.")
pred_df = pd.read_csv(pred_file)
y_test = pd.to_numeric(pred_df["y_true"], errors='coerce').dropna().values
y_pred = pd.to_numeric(pred_df["y_pred"], errors='coerce').dropna().values

# 3. Load the trained XGBoost model for real feature importances
model_file = f"xgb_model_{target_var}.json"
if not os.path.exists(model_file):
    raise FileNotFoundError(f"{model_file} not found. Run kmeans_xgboost_train.py first.")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(model_file)
band_align_cols = [c for c in df.columns if c.startswith("band_align_")]
feature_cols = [c for c in [
    "dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
    "Nt_cm3", "eps_shell", "Vmax_V", "lattice_mismatch_pct",
] if c in df.columns] + band_align_cols
scores = xgb_model.feature_importances_
feat_imp = (pd.DataFrame({'feature': feature_cols[:len(scores)], 'importance': scores})
            .sort_values('importance', ascending=False).head(8))

r2_val = r2_score(y_test, y_pred) if len(y_test) > 0 else 0.0
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred)) if len(y_test) > 0 else 0.0
residuals = y_test - y_pred


# ==========================================
# IMAGE 1: Model Accuracy / Parity Plot
# ==========================================
fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=300)
ax1.scatter(y_test, y_pred, color='#3B82F6', edgecolors='#1E3A8A', alpha=0.7, s=40, label='Test Candidates')
if len(y_test) > 0:
    min_v, max_v = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax1.plot([min_v, max_v], [min_v, max_v], color='#EF4444', linestyle='--', linewidth=1.8, label='Ideal Parity (1:1)')
ax1.set_title(f'Model Accuracy ($R^2 = {r2_val:.3f}$, RMSE = {rmse_val:.3f})', fontweight='bold', fontsize=11)
ax1.set_xlabel('Actual Value', fontweight='bold', fontsize=10)
ax1.set_ylabel('Predicted Value', fontweight='bold', fontsize=10)
ax1.legend(frameon=True, facecolor='white', loc='upper left')
ax1.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig1.savefig("plots/plot_1_parity.png", dpi=300, bbox_inches='tight')
plt.close(fig1)


# ==========================================
# IMAGE 2: Residual Error Distribution
# ==========================================
fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=300)
n, bins, _ = ax2.hist(residuals, bins=15, color='#F43F5E', edgecolor='black', alpha=0.7, density=True)
if len(residuals) > 0:
    mu, std = norm.fit(residuals)
    x_axis = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x_axis, norm.pdf(x_axis, mu, std), color='#9F1239', linewidth=2, label='Gaussian Fit')
ax2.axvline(0, color='black', linestyle='--', alpha=0.7)
ax2.set_title('Error Distribution (Residuals)', fontweight='bold', fontsize=11)
ax2.set_xlabel(r'Error ($y_{\mathrm{true}} - y_{\mathrm{pred}}$)', fontweight='bold', fontsize=10)
ax2.set_ylabel('Density', fontweight='bold', fontsize=10)
ax2.legend(frameon=True, facecolor='white')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig2.savefig("plots/plot_2_residuals.png", dpi=300, bbox_inches='tight')
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
fig3.savefig("plots/plot_3_screening.png", dpi=300, bbox_inches='tight')
plt.close(fig3)


# ==========================================
# IMAGE 4: Top Model Feature Importances
# ==========================================
fig4, ax4 = plt.subplots(figsize=(6, 5), dpi=300)
palette = sns.color_palette("mako", n_colors=len(feat_imp))
ax4.barh(feat_imp['feature'][::-1], feat_imp['importance'][::-1], color=palette[::-1], edgecolor='black', linewidth=0.5)
ax4.set_title(f'XGBoost Feature Importance: {target_var}', fontweight='bold', fontsize=11)
ax4.set_xlabel('Gain-based importance', fontweight='bold', fontsize=10)
ax4.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
fig4.savefig("plots/plot_4_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close(fig4)

print("Successfully generated and saved 4 individual PNG plots:")
print(" - plots/plot_1_parity.png")
print(" - plots/plot_2_residuals.png")
print(" - plots/plot_3_screening.png")
print(" - plots/plot_4_feature_importance.png")