# ============================================================
#  💰 SALES PREDICTION USING PYTHON — ML INTERNSHIP PROJECT
#  Author  : Machine Learning Internship
#  Dataset : Advertising Dataset (200 samples)
#  Target  : Sales (in thousands of units)
#  Features: TV, Radio, Newspaper advertising budgets ($000s)
#  Models  : Linear Regression, Ridge, Lasso, ElasticNet,
#             Decision Tree, Random Forest, SVR, Gradient Boost
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing    import StandardScaler, PolynomialFeatures
from sklearn.linear_model     import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree             import DecisionTreeRegressor
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm              import SVR
from sklearn.metrics          import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.inspection       import permutation_importance
import os

# ─────────────────────────────────────────────
# 0.  OUTPUT FOLDER
# ─────────────────────────────────────────────
OUT = "."
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
print("=" * 65)
print("  💰  SALES PREDICTION — ADVERTISING SPEND vs SALES")
print("=" * 65)

df = pd.read_csv("/Users/aryansisodiya/Downloads/Advertising.csv.xls")
df.drop(columns=["Unnamed: 0"], inplace=True)

features = ["TV", "Radio", "Newspaper"]
target   = "Sales"

print(f"\n📊 Dataset Shape  : {df.shape}")
print(f"📋 Features       : {features}")
print(f"🎯 Target         : {target} (units in thousands)")
print(f"\n{df.head(8).to_string(index=False)}")

# ─────────────────────────────────────────────
# 2.  EDA
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  📈  EXPLORATORY DATA ANALYSIS")
print("─" * 65)

print(f"\n🔍 Missing Values  : {df.isnull().sum().sum()}")
print(f"\n📐 Descriptive Statistics:")
print(df.describe().round(2).to_string())

# ── EDA Figure ──────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle("Sales Prediction — Exploratory Data Analysis",
             fontsize=16, fontweight="bold", y=0.98)

PALETTE = {"TV": "#E91E63", "Radio": "#2196F3", "Newspaper": "#FF9800"}

# (1) Scatter: each feature vs Sales
for idx, feat in enumerate(features):
    ax = fig.add_subplot(4, 3, idx + 1)
    ax.scatter(df[feat], df[target],
               color=list(PALETTE.values())[idx], alpha=0.7, s=50, edgecolors="white")
    # regression line
    m, b = np.polyfit(df[feat], df[target], 1)
    xs = np.linspace(df[feat].min(), df[feat].max(), 100)
    ax.plot(xs, m*xs + b, color="black", linewidth=2, linestyle="--")
    ax.set_xlabel(f"{feat} Budget ($000s)")
    ax.set_ylabel("Sales (000 units)")
    ax.set_title(f"{feat} vs Sales  (r={df[feat].corr(df[target]):.2f})")
    ax.grid(alpha=0.3)

# (2) Distribution of Sales
ax = fig.add_subplot(4, 3, 4)
ax.hist(df[target], bins=20, color="#4CAF50", edgecolor="white", alpha=0.85)
ax.axvline(df[target].mean(), color="red", linestyle="--", linewidth=2,
           label=f"Mean={df[target].mean():.1f}")
ax.axvline(df[target].median(), color="blue", linestyle=":", linewidth=2,
           label=f"Median={df[target].median():.1f}")
ax.set_xlabel("Sales (000 units)")
ax.set_ylabel("Count")
ax.set_title("Sales Distribution")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# (3) Distribution of each feature
for idx, feat in enumerate(features):
    ax = fig.add_subplot(4, 3, 5 + idx)
    ax.hist(df[feat], bins=20,
            color=list(PALETTE.values())[idx], edgecolor="white", alpha=0.85)
    ax.axvline(df[feat].mean(), color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel(f"{feat} Budget ($000s)")
    ax.set_ylabel("Count")
    ax.set_title(f"{feat} Distribution")
    ax.grid(alpha=0.3)

# (4) Correlation heatmap
ax = fig.add_subplot(4, 3, 8)
corr = df.corr()
mask = np.zeros_like(corr)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1, mask=mask)
ax.set_title("Correlation Matrix")

# (5) Box plots
ax = fig.add_subplot(4, 3, 9)
df_melt = df.melt(value_vars=features, var_name="Channel", value_name="Budget")
palette = {"TV": "#E91E63", "Radio": "#2196F3", "Newspaper": "#FF9800"}
sns.boxplot(data=df_melt, x="Channel", y="Budget", palette=palette, ax=ax)
ax.set_title("Ad Budget Distribution by Channel")
ax.set_ylabel("Budget ($000s)")
ax.grid(axis="y", alpha=0.3)

# (6) Pairplot-style: TV vs Radio coloured by Sales
ax = fig.add_subplot(4, 3, 10)
sc = ax.scatter(df["TV"], df["Radio"], c=df["Sales"],
                cmap="RdYlGn", s=60, edgecolors="white", alpha=0.9)
plt.colorbar(sc, ax=ax, label="Sales")
ax.set_xlabel("TV Budget ($000s)")
ax.set_ylabel("Radio Budget ($000s)")
ax.set_title("TV vs Radio (colour = Sales)")
ax.grid(alpha=0.3)

# (7) Cumulative ad spend vs Sales
ax = fig.add_subplot(4, 3, 11)
df["Total_Spend"] = df[features].sum(axis=1)
ax.scatter(df["Total_Spend"], df["Sales"],
           color="#9C27B0", alpha=0.7, s=50, edgecolors="white")
m, b = np.polyfit(df["Total_Spend"], df["Sales"], 1)
xs = np.linspace(df["Total_Spend"].min(), df["Total_Spend"].max(), 100)
ax.plot(xs, m*xs + b, color="black", linewidth=2, linestyle="--")
ax.set_xlabel("Total Ad Spend ($000s)")
ax.set_ylabel("Sales (000 units)")
ax.set_title(f"Total Spend vs Sales  (r={df['Total_Spend'].corr(df[target]):.2f})")
ax.grid(alpha=0.3)
df.drop(columns=["Total_Spend"], inplace=True)

# (8) Sales vs TV — highlight high/low Radio
ax = fig.add_subplot(4, 3, 12)
high_radio = df["Radio"] > df["Radio"].median()
ax.scatter(df.loc[~high_radio, "TV"],  df.loc[~high_radio,  "Sales"],
           color="#90CAF9", alpha=0.7, s=50, label="Low Radio", edgecolors="white")
ax.scatter(df.loc[high_radio,  "TV"],  df.loc[high_radio,   "Sales"],
           color="#F44336", alpha=0.7, s=50, label="High Radio", edgecolors="white")
ax.set_xlabel("TV Budget ($000s)")
ax.set_ylabel("Sales (000 units)")
ax.set_title("TV vs Sales (by Radio level)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT}/1_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅  EDA plots saved.")

# ─────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  ⚙️   PREPROCESSING")
print("─" * 65)

X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler   = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"  Train size    : {len(X_train)} samples")
print(f"  Test  size    : {len(X_test)}  samples")
print(f"  Scaling       : StandardScaler applied")

# ─────────────────────────────────────────────
# 4.  TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  🤖  TRAINING 8 REGRESSION MODELS")
print("─" * 65)

models = {
    "Linear Regression"     : LinearRegression(),
    "Ridge Regression"      : Ridge(alpha=1.0),
    "Lasso Regression"      : Lasso(alpha=0.1),
    "ElasticNet"            : ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Decision Tree"         : DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest"         : RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector (SVR)"  : SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
    "Gradient Boosting"     : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

def evaluate(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae  = mean_absolute_error(yte, pred)
    mse  = mean_squared_error(yte, pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(yte, pred)
    cv   = cross_val_score(model, Xtr, ytr, cv=5, scoring="r2")
    return {
        "model": model, "pred": pred,
        "MAE": mae, "MSE": mse, "RMSE": rmse,
        "R²": r2, "CV_R²_mean": cv.mean(), "CV_R²_std": cv.std()
    }

results = {}
print(f"\n  {'Model':<26} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'CV R²':>8} {'CV Std':>8}")
print("  " + "─" * 68)

for name, model in models.items():
    res = evaluate(name, model, X_train_s, X_test_s, y_train, y_test)
    results[name] = res
    print(f"  {name:<26} {res['MAE']:>7.4f} {res['RMSE']:>7.4f} "
          f"{res['R²']:>7.4f} {res['CV_R²_mean']:>8.4f} {res['CV_R²_std']:>8.4f}")

# ─────────────────────────────────────────────
# 5.  BEST MODEL
# ─────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["R²"])
best       = results[best_name]
best_model = best["model"]
y_pred     = best["pred"]

print("\n" + "─" * 65)
print(f"  🏆  BEST MODEL  :  {best_name}")
print(f"      R² Score    :  {best['R²']:.4f}  ({best['R²']*100:.2f}%  variance explained)")
print(f"      MAE         :  {best['MAE']:.4f}  (${best['MAE']*1000:.0f} avg error)")
print(f"      RMSE        :  {best['RMSE']:.4f}")
print("─" * 65)

# ─────────────────────────────────────────────
# 6.  POLYNOMIAL REGRESSION (bonus)
# ─────────────────────────────────────────────
print("\n  📐 Polynomial Regression (degree=2) ...")
poly     = PolynomialFeatures(degree=2, include_bias=False)
X_poly_tr = poly.fit_transform(X_train_s)
X_poly_te = poly.transform(X_test_s)
lr_poly   = LinearRegression()
lr_poly.fit(X_poly_tr, y_train)
y_poly_pred = lr_poly.predict(X_poly_te)
poly_r2  = r2_score(y_test, y_poly_pred)
poly_mae = mean_absolute_error(y_test, y_poly_pred)
poly_rmse= np.sqrt(mean_squared_error(y_test, y_poly_pred))
print(f"  Polynomial R²   : {poly_r2:.4f}  MAE: {poly_mae:.4f}  RMSE: {poly_rmse:.4f}")

# ─────────────────────────────────────────────
# 7.  VISUALISATIONS
# ─────────────────────────────────────────────

# ── 7a  Model Comparison ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Comparison — Regression Metrics", fontsize=14, fontweight="bold")

names_s = [n.replace(" ", "\n") for n in results.keys()]
x       = np.arange(len(results))
w       = 0.6

for ax, metric, color, title in zip(
    axes,
    ["R²", "RMSE", "MAE"],
    ["#4CAF50", "#F44336", "#FF9800"],
    ["R² Score (higher = better)", "RMSE (lower = better)", "MAE (lower = better)"]
):
    vals = [results[n][metric] for n in results]
    bars = ax.bar(x, vals, width=w, color=color, edgecolor="white", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names_s, fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    # highlight best
    best_idx = (vals.index(max(vals)) if metric == "R²"
                else vals.index(min(vals)))
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)

plt.tight_layout()
fig.savefig(f"{OUT}/2_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 7b  Actual vs Predicted (all models) ──────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Actual vs Predicted Sales — All Models", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes.flatten(), results.items()):
    ax.scatter(y_test, res["pred"], color="#2196F3",
               alpha=0.7, s=50, edgecolors="white")
    lims = [min(y_test.min(), res["pred"].min()) - 1,
            max(y_test.max(), res["pred"].max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title(f"{name}\nR²={res['R²']:.3f}  MAE={res['MAE']:.3f}", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT}/3_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 7c  Residual Analysis (Best Model) ────────
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Residual Analysis — {best_name}", fontsize=13, fontweight="bold")

# residual scatter
ax = axes[0]
ax.scatter(y_pred, residuals, color="#9C27B0", alpha=0.75, s=55, edgecolors="white")
ax.axhline(0, color="red", linewidth=2, linestyle="--")
ax.set_xlabel("Predicted Sales")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Predicted")
ax.grid(alpha=0.3)

# residual histogram
ax = axes[1]
ax.hist(residuals, bins=18, color="#009688", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", linewidth=2, linestyle="--")
ax.set_xlabel("Residual Value")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution")
ax.grid(alpha=0.3)

# Q-Q plot
ax = axes[2]
from scipy import stats
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, fit=True)
ax.scatter(osm, osr, color="#FF5722", alpha=0.75, s=50, edgecolors="white")
xs = np.array([min(osm), max(osm)])
ax.plot(xs, slope*xs + intercept, "b--", linewidth=2)
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.set_title("Q-Q Plot (Normality Check)")
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT}/4_residual_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 7d  Feature Importance ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Feature Importance & Coefficients", fontsize=13, fontweight="bold")

# Linear Regression coefficients
lr      = results["Linear Regression"]["model"]
coefs   = np.abs(lr.coef_)
colors  = ["#E91E63", "#2196F3", "#FF9800"]
ax = axes[0]
bars = ax.bar(features, lr.coef_, color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Linear Regression Coefficients")
ax.set_ylabel("Coefficient Value")
ax.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, lr.coef_):
    ax.text(bar.get_x() + bar.get_width()/2,
            v + (0.05 if v >= 0 else -0.15),
            f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Random Forest feature importance
rf = results["Random Forest"]["model"]
fi = rf.feature_importances_
ax = axes[1]
sorted_idx = np.argsort(fi)[::-1]
ax.bar([features[i] for i in sorted_idx],
       fi[sorted_idx], color=[colors[i] for i in sorted_idx], edgecolor="white")
ax.set_title("Random Forest Feature Importance")
ax.set_ylabel("Importance Score")
ax.grid(axis="y", alpha=0.3)
for i, (feat, val) in enumerate(zip([features[j] for j in sorted_idx], fi[sorted_idx])):
    ax.text(i, val + 0.005, f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
fig.savefig(f"{OUT}/5_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 7e  Sales Prediction Surface (TV + Radio) ─
fig = plt.figure(figsize=(15, 5))
fig.suptitle("Sales Prediction Surface — TV & Radio Budget", fontsize=13, fontweight="bold")

tv_range    = np.linspace(df["TV"].min(),    df["TV"].max(),    50)
radio_range = np.linspace(df["Radio"].min(), df["Radio"].max(), 50)
TV_grid, Radio_grid = np.meshgrid(tv_range, radio_range)

# fix Newspaper at its mean
news_mean = df["Newspaper"].mean()
grid_pts  = np.column_stack([
    TV_grid.ravel(),
    Radio_grid.ravel(),
    np.full(TV_grid.size, news_mean)
])
grid_scaled   = scaler.transform(grid_pts)
sales_surface = best_model.predict(grid_scaled).reshape(TV_grid.shape)

# 3D Surface
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
surf = ax1.plot_surface(TV_grid, Radio_grid, sales_surface,
                        cmap="RdYlGn", alpha=0.85, edgecolor="none")
fig.colorbar(surf, ax=ax1, shrink=0.5, label="Pred. Sales")
ax1.set_xlabel("TV ($000s)")
ax1.set_ylabel("Radio ($000s)")
ax1.set_zlabel("Sales")
ax1.set_title(f"3D Surface\n({best_name})")

# Contour
ax2 = fig.add_subplot(1, 3, 2)
ct = ax2.contourf(TV_grid, Radio_grid, sales_surface, 20, cmap="RdYlGn")
plt.colorbar(ct, ax=ax2, label="Predicted Sales")
ax2.scatter(df["TV"], df["Radio"], c=df["Sales"],
            cmap="RdYlGn", s=30, edgecolors="black", linewidth=0.3)
ax2.set_xlabel("TV Budget ($000s)")
ax2.set_ylabel("Radio Budget ($000s)")
ax2.set_title("Contour Map")

# Cross-val R² comparison
ax3 = fig.add_subplot(1, 3, 3)
cv_means = [results[n]["CV_R²_mean"] for n in results]
cv_stds  = [results[n]["CV_R²_std"]  for n in results]
short_n  = [n.replace(" ", "\n") for n in results]
bars = ax3.bar(range(len(results)), cv_means, color="#3F51B5",
               edgecolor="white", alpha=0.85)
ax3.errorbar(range(len(results)), cv_means, yerr=cv_stds,
             fmt="none", capsize=4, color="black", linewidth=1.5)
ax3.set_xticks(range(len(results)))
ax3.set_xticklabels(short_n, fontsize=7)
ax3.set_ylim(0.7, 1.02)
ax3.set_title("5-Fold CV R² Score")
ax3.set_ylabel("CV R²")
ax3.grid(axis="y", alpha=0.3)
for bar, v in zip(bars, cv_means):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.006, f"{v:.3f}",
             ha="center", va="bottom", fontsize=7)

plt.tight_layout()
fig.savefig(f"{OUT}/6_prediction_surface.png", dpi=150, bbox_inches="tight")
plt.close()

print("✅  All visualisation plots saved.\n")

# ─────────────────────────────────────────────
# 8.  HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
print("─" * 65)
print(f"  🔧  HYPERPARAMETER TUNING — {best_name}")
print("─" * 65)

param_grids = {
    "Linear Regression"    : {"fit_intercept": [True, False]},
    "Ridge Regression"     : {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso Regression"     : {"alpha": [0.001, 0.01, 0.1, 1, 10]},
    "ElasticNet"           : {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]},
    "Decision Tree"        : {"max_depth": [3, 4, 5, 6, None], "min_samples_split": [2, 5, 10]},
    "Random Forest"        : {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    "Support Vector (SVR)" : {"C": [1, 10, 100], "gamma": [0.01, 0.1, 1]},
    "Gradient Boosting"    : {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2]},
}

grid = GridSearchCV(
    best_model, param_grids[best_name],
    cv=5, scoring="r2", n_jobs=-1
)
grid.fit(X_train_s, y_train)
tuned_pred = grid.best_estimator_.predict(X_test_s)
tuned_r2   = r2_score(y_test, tuned_pred)
tuned_mae  = mean_absolute_error(y_test, tuned_pred)

print(f"  Best Params : {grid.best_params_}")
print(f"  Tuned R²    : {tuned_r2:.4f}   (before: {best['R²']:.4f})")
print(f"  Tuned MAE   : {tuned_mae:.4f}   (before: {best['MAE']:.4f})")

# ─────────────────────────────────────────────
# 9.  FINAL SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  📊  FINAL MODEL SUMMARY")
print("=" * 65)

summary = pd.DataFrame({
    "Model"    : list(results.keys()),
    "R²"       : [f"{results[n]['R²']:.4f}"        for n in results],
    "MAE"      : [f"{results[n]['MAE']:.4f}"        for n in results],
    "RMSE"     : [f"{results[n]['RMSE']:.4f}"       for n in results],
    "CV R²"    : [f"{results[n]['CV_R²_mean']:.4f}" for n in results],
    "CV Std"   : [f"{results[n]['CV_R²_std']:.4f}"  for n in results],
})
print(summary.to_string(index=False))
summary.to_csv(f"{OUT}/model_summary.csv", index=False)

# ─────────────────────────────────────────────
# 10.  BUSINESS INSIGHT — ROI ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  💡  BUSINESS INSIGHT — AD SPEND vs SALES IMPACT")
print("─" * 65)

lr = results["Linear Regression"]["model"]
print(f"\n  Linear Regression Coefficients (per $1000 increase in budget):")
print(f"  {'Channel':<15} {'Coefficient':>14} {'Impact':>25}")
print("  " + "─" * 58)
for feat, coef in zip(features, lr.coef_):
    print(f"  {feat:<15} {coef:>14.4f}  → +{coef:.2f}K units per $1K spend")
print(f"\n  📌 Key Insight: TV and Radio have the strongest impact on Sales.")
print(f"     Newspaper advertising has minimal effect.")
print(f"     Recommendation: Allocate MORE budget to TV & Radio campaigns.")

# ─────────────────────────────────────────────
# 11.  LIVE SALES PREDICTION DEMO
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  🌟  LIVE SALES PREDICTION DEMO")
print("─" * 65)

scenarios = [
    [230.1, 37.8, 69.2,  "High TV, Medium Radio"],
    [44.5,  39.3, 45.1,  "Low TV, High Radio"],
    [100.0, 50.0, 10.0,  "Medium TV, Max Radio"],
    [0.7,   39.6, 8.7,   "No TV, Radio focused"],
    [296.4, 49.6, 114.0, "Maximum Budget ALL"],
    [50.0,  10.0, 5.0,   "Minimal Budget"],
]

tuned_model = grid.best_estimator_
print(f"\n  Using Best Model: {best_name}")
print(f"  {'TV':>6} {'Radio':>7} {'News':>7} | {'Predicted Sales':>17} | Strategy")
print("  " + "─" * 70)
for sc_row in scenarios:
    tv, radio, news, label = sc_row
    inp    = np.array([[tv, radio, news]])
    inp_sc = scaler.transform(inp)
    pred   = tuned_model.predict(inp_sc)[0]
    print(f"  {tv:>6.1f} {radio:>7.1f} {news:>7.1f} | "
          f"{pred:>12.2f}K units | {label}")

print("\n" + "=" * 65)
print("  ✅  PROJECT COMPLETE")
print(f"  All files saved to: {OUT}")
print("=" * 65)
