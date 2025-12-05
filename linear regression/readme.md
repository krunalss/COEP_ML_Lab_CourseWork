# Assignment 1 Outputs
## (Diabetes Linear Regression – Model)
- Experimented with three linear models – Linear Regression, Ridge, and Lasso – to identify the best performer on the diabetes dataset.  
- Applied GridSearchCV with cross-validation to tune key hyperparameters (mainly α for Ridge and Lasso) instead of using default values.  
- Evaluated all models using MSE, RMSE, MAE, and R², and selected the final model based on lowest test RMSE and higher R². 
- Created visual comparisons (bar plots and regression lines) to clearly show how each enhancement affected model performance and generalization.  


## 1. All Features in the Dataset
_All input feature names (age, sex, bmi, bp, s1–s6) and the target column from the diabetes dataset._
![All features](allss/All_features.png)  

---

## 2. DataFrame Info
_Structure of the DataFrame showing 442 rows, 10 standardized feature columns, and the target column with no missing values._
![df.info()](allss/df_info.png)  


---

## 3. Descriptive Statistics
_Summary statistics (count, mean, std, min, max, quartiles) for each standardized feature and the target._

![df.describe()](allss/df_describe.png)  

---

## 4. Base Model Metrics
_Table of performance metrics (MSE, RMSE, MAE, R²) for the base Linear, Ridge, and Lasso regression models._  
![results](allss/results.png)  


---

## 5. RMSE Comparison – Base Models
_Bar chart comparing RMSE values of the base Linear, Ridge, and Lasso models (lower RMSE indicates better performance)._
![RMSE Comparison](allss/RMSE_Comparison.png)  


---

## 6. Tuned Model Metrics
_Table of performance metrics for the hyperparameter-tuned Linear, Ridge, and Lasso regression models._
![results_tuned](allss/results_tuned.png)  


---

## 7. RMSE – Base vs Tuned Models
_Grouped bar chart comparing RMSE of base vs hyperparameter-tuned models for Linear, Ridge, and Lasso regression._
![RMSE Base vs Tuned Models](allss/RMSE_Base_vs_Tuned_Models.png)  

