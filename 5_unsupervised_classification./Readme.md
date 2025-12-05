# Assignment 5 Outputs
## (Digits Dataset - Unsupervised Classification)
## Problem Statement
- Pickup any dataset from scikit-learn like Digits, Winedataset and perform the task of unsupervised classification.
  - a. Apply data pre-processing
  - b. Apply different unsupervised methods like K-means clustering, PCA etc.
  - c. Visualize the results using matplotlib/ seaborn. Show correlation matrix.

---

### **1. Dataset Shape**
![Dataset Shape](images/Screenshot%202025-12-05%20224652.png)
**Observation:** Dataset contains 1797 samples and 64 pixel features representing handwritten digits.

---

### **2. Correlation Matrix**
![Correlation Matrix](images/Screenshot%202025-12-05%20224737.png)
**Observation:** Adjacent pixels show higher correlation, indicating structured spatial patterns in digit images.

---

### **3. PCA Projection (True Labels)**
![PCA True Labels](images/Screenshot%202025-12-05%20224749.png)
**Observation:** Some digits naturally form separable clusters, while others overlap in PCA space.

---

### **4. Base K-Means Clusters**
![Base KMeans PCA](images/Screenshot%202025-12-05%20224758.png)
**Observation:** Base K-Means forms visible clusters but shows moderate overlap, indicating suboptimal separation.

---

### **5. Base K-Means Cluster Centers**
![Base Cluster Centers](images/Screenshot%202025-12-05%20224811.png)
**Observation:** Cluster centers resemble blurred digit shapes, showing the model captures general digit patterns.

---

### **6. Tuned K-Means Clusters**
![Tuned KMeans PCA](images/Screenshot%202025-12-05%20225746.png)
**Observation:** Tuned K-Means produces tighter and more distinct clusters compared to the base model.

---

### **7. Tuned K-Means Cluster Centers**
![Tuned Cluster Centers](images/Screenshot%202025-12-05%20225759.png)
**Observation:** Tuned centers are cleaner and more digit-like, indicating improved learning of digit prototypes.

---

