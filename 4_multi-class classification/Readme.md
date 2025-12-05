# Assignment 4 Outputs
## (Iris -multi-class classification)
## Problem Statement
4. Pickup any dataset from scikit-learn like Iris dataset and solve the problem of multi-class classification.  
- a. Apply data pre-processing  
- b. Show the results with different methods  
  - i. Logistic regression  
  - ii. KNN  
  - iii. Decision tree  
  - iv. SVC  
- c. Evaluate using different measures. Make sure the final result is having the least error.  
- d. Visualize the results using matplotlib/ seaborn. Show correlation matrix.  
## 1. First Data Preview (Head of Dataset + Target)
![Dataset Head](images/Screenshot%202025-12-05%20220116.png)  

**Observation:** All shown samples belong to class 0.

## 2. Logistic Regression – Classification Report
![LR Report](images/Screenshot%202025-12-05%20220128.png)  

**Observation:** Achieves 0.93 accuracy with balanced precision and recall across classes.

## 3. KNN – Classification Report
![KNN Report](images/Screenshot%202025-12-05%20220143.png)  

**Observation:** Accuracy remains 0.93; recall slightly drops for class 2.

## 4. Decision Tree – Classification Report
![DT Report](images/Screenshot%202025-12-05%20220151.png)  

**Observation:** Accuracy is 0.93 with uniform class performance.

## 5. SVC – Classification Report
![SVC Report](images/Screenshot%202025-12-05%20220157.png)  

**Observation:** Highest accuracy (0.97) among all evaluated models.

## 6. Logistic Regression – Confusion Matrix
![LR CM](images/Screenshot%202025-12-05%20220217.png)  

**Observation:** One misclassification observed in class 2.

## 7. KNN – Confusion Matrix
![KNN CM](images/Screenshot%202025-12-05%20220239.png)  

**Observation:** Two misclassifications in class 2.

## 8. Decision Tree – Confusion Matrix
![DT CM](images/Screenshot%202025-12-05%20220255.png)  

**Observation:** One misclassification in class 2.

## 9. SVC – Confusion Matrix
![SVC CM](images/Screenshot%202025-12-05%20220323.png)  

**Observation:** Only one misclassification; best performance overall.

## 10. Correlation Matrix of Iris Dataset
![Correlation Matrix](images/Screenshot%202025-12-05%20220344.png)  

**Observation:** Petal length and petal width have strongest correlation (~0.96).

## 11. Accuracy Comparison – Full Scale Bar Graph
![Full Bar Graph](images/Screenshot%202025-12-05%20220447.png)  

**Observation:** SVC (Base) achieves the highest accuracy, followed by tuned SVC.

## 12. Accuracy Comparison – Zoomed-In Bar Graph
![Zoomed Bar Graph](images/Screenshot%202025-12-05%20220458.png)  

**Observation:** SVC (Base) clearly stands out with the highest accuracy (~0.97) so parameter tunning of SVC doesnt not help to betterment of accuracy.
