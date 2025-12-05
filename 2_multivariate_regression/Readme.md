# Assignment 2 Outputs
## (Linnerud dataset - Polynomial Linear Regression–Model)

### Feature & Target Columns (Initial Preview)
- Confirms which variables are used as **inputs** and which as **outputs**.
![Feature and target columns](allss/Screenshot%202025-12-05%20194819.png)




---

## Dataset Shape & Summary Statistics
- Confirms there are **no missing values**.
- Gives an overview of the **range and distribution** of each feature and target.
- Helps understand typical values and variability before modeling.

![Dataset shape and summary statistics](allss/Screenshot%202025-12-05%20194842.png)

## Features (X) and Targets (Y) Tables
- Confirms the input matrix `X` and target matrix `Y` have aligned rows and correct values.
![Features and targets tables](allss/Screenshot%202025-12-05%20194850.png)



---

## Correlation Matrix & Feature–Weight Correlations
- Quantifies how strongly each exercise variable is **linearly related** to `Weight`.

![Correlation matrix and feature-weight correlations](allss/Screenshot%202025-12-05%20194906.png)


---

## Bar Plot – Linnerud Feature–Weight Correlations
- Provides a **visual comparison** of the correlation strength of each feature with `Weight`.
![Bar chart of feature-weight correlations](allss/Screenshot%202025-12-05%20194926.png)

---

## Selected Features & X_subset
- Confirms that **all three features** are selected based on their correlation magnitude.
![Selected features and X_subset](allss/Screenshot%202025-12-05%20194941.png)

---

## Baseline Linear Regression (Subset → Weight)
- Shows performance of a **simple multiple linear regression** model using the selected features to predict `Weight`.
![Baseline linear regression metrics](allss/Screenshot%202025-12-05%20194949.png)



---

## Polynomial Regression (degree = 2, Subset → Weight)
- Demonstrates the effect of adding **degree-2 polynomial features**.
- The model fits the training data extremely well (high R²), but performs **worse on the test set**, showing classic **overfitting** due to high complexity on a small dataset.

![Polynomial regression metrics](allss/Screenshot%202025-12-05%20194955.png)

---

## 9️⃣ Multi-output Linear Regression (Weight & Waist)
- Shows a true **multi-output (multivariate) regression** model predicting both `Weight` and `Waist` together.
![Multi-output regression metrics](allss/Screenshot%202025-12-05%20195007.png)





---

## Hyperparameter Tuning – Baseline Linear (Ridge)
- Summarizes the **GridSearchCV** for Ridge regression applied to the baseline model.
- Indicates that **α = 100.0** is the best regularization strength (among tested values) in terms of cross-validated MSE.  
![Hyperparameter tuning baseline ridge](allss/Screenshot%202025-12-05%20195014.png)

---

## Hyperparameter Tuning – Polynomial + Ridge
- In this run, the best degree is actually **1**, which effectively falls back to a **linear model with Ridge**, and α = 10.0.
![Hyperparameter tuning: Polynomial + Ridge](allss/Screenshot%202025-12-05%20195022.png)



---

## Hyperparameter Tuning – Multi-output Ridge (Weight & Waist)
- Shows the result of hyperparameter tuning for a **multi-output Ridge regression** that predicts both `Weight` and `Waist`.
- Identifies **α = 10.0** as the best regularization strength for the joint model.
![Hyperparameter tuning: Multi-output Ridge](allss/Screenshot%202025-12-05%20195033.png)


---

## Grouped Bar Chart – RMSE: Base vs Tuned Models
- Provides a **visual comparison** of how RMSE changes after hyperparameter tuning:
  - Tuned models generally show **lower RMSE** than their base versions.
  - You can quickly see which models benefited most from tuning (e.g., large drop for polynomial and multi-output Weight).
![RMSE comparison – Base vs Tuned models](allss/Screenshot%202025-12-05%20195045.png)

---

## Regression Line Plot – Tuned Baseline Linear (Weight vs Situps)
- Visualizes how the **tuned baseline linear model** relates `Situps` to `Weight`.
- Shows a **slightly negative slope**, consistent with the negative correlation between `Situps` and `Weight`.
- Helps interpret the model in a more intuitive way:
  - As the number of `Situps` increases, predicted `Weight` tends to decrease slightly, according to the tuned model.
![Regression line of tuned Baseline Linear model](allss/Screenshot%202025-12-05%20195055.png)



---
