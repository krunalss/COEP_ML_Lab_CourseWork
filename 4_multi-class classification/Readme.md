# Output.md

## 1. First Data Preview (Head of Dataset + Target)
![Dataset Head](images/Screenshot%202025-12-05%20220116.png)  
**Description:** Displays first five rows of Iris dataset features and target values.  
**Observation:** All shown samples belong to class 0.

## 2. Logistic Regression – Classification Report
![LR Report](images/Screenshot%202025-12-05%20220128.png)  
**Description:** Performance metrics of Logistic Regression model.  
**Observation:** Achieves 0.93 accuracy with balanced precision and recall across classes.

## 3. KNN – Classification Report
![KNN Report](images/Screenshot%202025-12-05%20220143.png)  
**Description:** Evaluation results of KNN classifier.  
**Observation:** Accuracy remains 0.93; recall slightly drops for class 2.

## 4. Decision Tree – Classification Report
![DT Report](images/Screenshot%202025-12-05%20220151.png)  
**Description:** Classification metrics for Decision Tree model.  
**Observation:** Accuracy is 0.93 with uniform class performance.

## 5. SVC – Classification Report
![SVC Report](images/Screenshot%202025-12-05%20220157.png)  
**Description:** SVC performance metrics including precision, recall, and f1-score.  
**Observation:** Highest accuracy (0.97) among all evaluated models.

## 6. Logistic Regression – Confusion Matrix
![LR CM](images/Screenshot%202025-12-05%20220217.png)  
**Description:** Confusion matrix for Logistic Regression predictions.  
**Observation:** One misclassification observed in class 2.

## 7. KNN – Confusion Matrix
![KNN CM](images/Screenshot%202025-12-05%20220239.png)  
**Description:** Confusion matrix visualization of KNN classifier.  
**Observation:** Two misclassifications in class 2.

## 8. Decision Tree – Confusion Matrix
![DT CM](images/Screenshot%202025-12-05%20220255.png)  
**Description:** Confusion matrix for Decision Tree model.  
**Observation:** One misclassification in class 2.

## 9. SVC – Confusion Matrix
![SVC CM](images/Screenshot%202025-12-05%20220323.png)  
**Description:** Confusion matrix of SVC model predictions.  
**Observation:** Only one misclassification; best performance overall.

## 10. Correlation Matrix of Iris Dataset
![Correlation Matrix](images/Screenshot%202025-12-05%20220344.png)  
**Description:** Heatmap showing correlations among Iris dataset features.  
**Observation:** Petal length and petal width have strongest correlation (~0.96).

## 11. Accuracy Comparison – Full Scale Bar Graph
![Full Bar Graph](images/Screenshot%202025-12-05%20220447.png)  
**Description:** Bar chart comparing accuracy of all models including tuned SVC.  
**Observation:** SVC (Base) achieves the highest accuracy, followed by tuned SVC.

## 12. Accuracy Comparison – Zoomed-In Bar Graph
![Zoomed Bar Graph](images/Screenshot%202025-12-05%20220458.png)  
**Description:** Zoomed accuracy comparison to highlight small differences among models.  
**Observation:** SVC (Base) clearly stands out with the highest accuracy (~0.97).
