

# Imports & Load Dataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


sns.set(style="whitegrid")

# Loading diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

#converting to dataframe
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("All features:\n", feature_names)
df.head()

df.info()

df.describe()

"""## Preprocessing (train-test split + scaling)"""

# Spliting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# noramlizing feature for Ridge/Lasso
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)

#Linear Regression (Vanilla)

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

y_pred_lin = lin_reg.predict(X_test_scaled)

mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
mae_lin = mean_absolute_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

print("=== Linear Regression (Vanilla) ===")
print(f"MSE  : {mse_lin:.4f}")
print(f"RMSE : {rmse_lin:.4f}")
print(f"MAE  : {mae_lin:.4f}")
print(f"R2   : {r2_lin:.4f}")

#Ridge Regression (L2 Regularization)

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("=== Ridge Regression (alpha = 1.0) ===")
print(f"MSE  : {mse_ridge:.4f}")
print(f"RMSE : {rmse_ridge:.4f}")
print(f"MAE  : {mae_ridge:.4f}")
print(f"R2   : {r2_ridge:.4f}")

#Lasso Regression (L1 Regularization)

lasso = Lasso(alpha=0.01, max_iter=10000, random_state=42)
lasso.fit(X_train_scaled, y_train)

y_pred_lasso = lasso.predict(X_test_scaled)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("=== Lasso Regression (alpha = 0.01) ===")
print(f"MSE  : {mse_lasso:.4f}")
print(f"RMSE : {rmse_lasso:.4f}")
print(f"MAE  : {mae_lasso:.4f}")
print(f"R2   : {r2_lasso:.4f}")

#Comparing and saving Linear vs Ridge vs Lasso using seaborn

results = pd.DataFrame([
    {
        "model": "Linear",
        "MSE": mse_lin,
        "RMSE": rmse_lin,
        "MAE": mae_lin,
        "R2": r2_lin,
    },
    {
        "model": "Ridge",
        "MSE": mse_ridge,
        "RMSE": rmse_ridge,
        "MAE": mae_ridge,
        "R2": r2_ridge,
    },
    {
        "model": "Lasso",
        "MSE": mse_lasso,
        "RMSE": rmse_lasso,
        "MAE": mae_lasso,
        "R2": r2_lasso,
    },
])

results

#Ploting RMSE comparison

plt.figure(figsize=(6, 4))
sns.barplot(data=results, x="model", y="RMSE")
plt.title("RMSE Comparison (Lower is Better)")
plt.tight_layout()
plt.show()

#Ploting R2 comparison

plt.figure(figsize=(6, 4))
sns.barplot(data=results, x="model", y="R2")
plt.title("R² Score Comparison (Higher is Better)")
plt.tight_layout()
plt.show()

#zooming the result so that difference will be visible for better comparision
results_df = pd.DataFrame(results)
results_df

plt.figure(figsize=(6, 4))
sns.barplot(data=results_df, x="model", y="RMSE")


rmse_min = results_df["RMSE"].min()
rmse_max = results_df["RMSE"].max()
plt.ylim(rmse_min - 0.2, rmse_max + 0.2)   # small margin around values

plt.title("RMSE Comparison (Zoomed In)")
plt.tight_layout()
plt.show()

"""**Hyperparameter** **tunning**"""

from sklearn.model_selection import GridSearchCV

#Hyperparameter tuning for Linear Regression

param_grid_lin = {
    "fit_intercept": [True, False],
    "positive": [False, True],
}

gs_lin = GridSearchCV(
    estimator=LinearRegression(),
    param_grid=param_grid_lin,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1,
)

gs_lin.fit(X_train_scaled, y_train)

lin_best = gs_lin.best_estimator_

print("Best params (Linear):", gs_lin.best_params_)
print("Best CV MSE (Linear):", -gs_lin.best_score_)

y_pred_lin_best = lin_best.predict(X_test_scaled)

mse_lin_best = mean_squared_error(y_test, y_pred_lin_best)
rmse_lin_best = np.sqrt(mse_lin_best)
mae_lin_best = mean_absolute_error(y_test, y_pred_lin_best)
r2_lin_best = r2_score(y_test, y_pred_lin_best)

print("\n=== Tuned Linear Regression (Test set) ===")
print(f"MSE  : {mse_lin_best:.4f}")
print(f"RMSE : {rmse_lin_best:.4f}")
print(f"MAE  : {mae_lin_best:.4f}")
print(f"R2   : {r2_lin_best:.4f}")

#Hyperparameter tuning for Ridge Regression

alpha_values_ridge = np.logspace(-3, 3, 20)

param_grid_ridge = {
    "alpha": alpha_values_ridge,
}

gs_ridge = GridSearchCV(
    estimator=Ridge(random_state=42),
    param_grid=param_grid_ridge,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1,
)

gs_ridge.fit(X_train_scaled, y_train)

ridge_best = gs_ridge.best_estimator_

print("Best params (Ridge):", gs_ridge.best_params_)
print("Best CV MSE (Ridge):", -gs_ridge.best_score_)

y_pred_ridge_best = ridge_best.predict(X_test_scaled)

mse_ridge_best = mean_squared_error(y_test, y_pred_ridge_best)
rmse_ridge_best = np.sqrt(mse_ridge_best)
mae_ridge_best = mean_absolute_error(y_test, y_pred_ridge_best)
r2_ridge_best = r2_score(y_test, y_pred_ridge_best)

print("\n=== Tuned Ridge Regression (Test set) ===")
print(f"MSE  : {mse_ridge_best:.4f}")
print(f"RMSE : {rmse_ridge_best:.4f}")
print(f"MAE  : {mae_ridge_best:.4f}")
print(f"R2   : {r2_ridge_best:.4f}")

#Hyperparameter tuning for Lasso Regression

alpha_values_lasso = np.logspace(-3, 1, 20)

param_grid_lasso = {
    "alpha": alpha_values_lasso,
}

gs_lasso = GridSearchCV(
    estimator=Lasso(max_iter=10000, random_state=42),
    param_grid=param_grid_lasso,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1,
)

gs_lasso.fit(X_train_scaled, y_train)

lasso_best = gs_lasso.best_estimator_

print("Best params (Lasso):", gs_lasso.best_params_)
print("Best CV MSE (Lasso):", -gs_lasso.best_score_)

y_pred_lasso_best = lasso_best.predict(X_test_scaled)

mse_lasso_best = mean_squared_error(y_test, y_pred_lasso_best)
rmse_lasso_best = np.sqrt(mse_lasso_best)
mae_lasso_best = mean_absolute_error(y_test, y_pred_lasso_best)
r2_lasso_best = r2_score(y_test, y_pred_lasso_best)

print("\n=== Tuned Lasso Regression (Test set) ===")
print(f"MSE  : {mse_lasso_best:.4f}")
print(f"RMSE : {rmse_lasso_best:.4f}")
print(f"MAE  : {mae_lasso_best:.4f}")
print(f"R2   : {r2_lasso_best:.4f}")

#Comparing tuned models

results_tuned = pd.DataFrame([
    {
        "model": "Linear (tuned)",
        "MSE": mse_lin_best,
        "RMSE": rmse_lin_best,
        "MAE": mae_lin_best,
        "R2": r2_lin_best,
    },
    {
        "model": "Ridge (tuned)",
        "MSE": mse_ridge_best,
        "RMSE": rmse_ridge_best,
        "MAE": mae_ridge_best,
        "R2": r2_ridge_best,
    },
    {
        "model": "Lasso (tuned)",
        "MSE": mse_lasso_best,
        "RMSE": rmse_lasso_best,
        "MAE": mae_lasso_best,
        "R2": r2_lasso_best,
    },
])

results_tuned

"""## Single bar chart with 6 bars (pairs per model)"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Make sure both are DataFrames
base  = pd.DataFrame(results).copy()
tuned = pd.DataFrame(results_tuned).copy()

# 2) Add version column
base["version"]  = "Base"
tuned["version"] = "Tuned"

# 3) Clean model names in tuned (remove ' (tuned)' so they match)
tuned["model"] = tuned["model"].str.replace(" (tuned)", "", regex=False)

# 4) Combine into one DataFrame (6 rows = 3 models × 2 versions)
results_combo = pd.concat([base, tuned], ignore_index=True)
print(results_combo)

# 5) Single grouped bar chart: 6 bars (pair per model)
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=results_combo,
    x="model",
    y="RMSE",        # change to "MSE", "MAE" or "R2" if you want
    hue="version"    # Base vs Tuned
)

# Optional: add value labels
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f"{h:.2f}",
                (p.get_x() + p.get_width() / 2, h),
                ha="center", va="bottom", fontsize=9)

# Optional: zoom slightly so differences are visible
rmse_min = results_combo["RMSE"].min()
rmse_max = results_combo["RMSE"].max()
plt.ylim(rmse_min - 0.5, rmse_max + 0.5)

plt.title("RMSE: Base vs Tuned Models")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()