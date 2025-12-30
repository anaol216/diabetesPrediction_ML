# COLLAGE NATURAL AND COMPUTATIONAL SCIENCE
# SCHOOL OF INFORMATION SCIENCE

**Title:** Optimal Route Finding for Taxi Services in Addis Ababa  
*(Applied Research: Diabetes Severity Classification Framework)*

### Group Members
| No. | NAME | ID |
|---|---|---|
| 1. | Anaol Atinafu | UGR/4751/15 |
| 2. | Bethel Baynesagn | UGR/3599/15 |
| 3. | Cherenet Kebede | UGR/9075/15 |
| 4. | Dandi Hirko | UGR/3573/15 |
| 5. | Nuniyat Getamesay | UGR/4059/15 |
| 6. | Rediet Mulugeta | UGR/4053/15 |

**Semester-Year:** 01/02 - 2025/26  
**Submission Date:** 26/12/2025

---

## 1. Introduction & Objective
The primary objective of this project is to implement a **Diabetes Severity Classification** system. While the foundational dataset provides continuous measures of disease progression, we transition this into a multiclass classification problem to assist in clinical decision-making. We utilize the **Decision Tree Classification** algorithm, a non-parametric supervised learning method, due to its inherent interpretability and ability to handle non-linear relationships through recursive partitioning.

## 2. Data Exploration
The study utilizes the Scikit-learn Diabetes dataset, which comprises:
- **Number of Samples:** 442 patients.
- **Features:** 10 physiological variables (Age, Sex, Body Mass Index, Average Blood Pressure, and six blood serum measurements).
- **Target Variable (Pre-discretization):** A quantitative measure of disease progression one year after baseline.

## 3. Methodology
### 3.1 Data Discretization
To perform classification, the continuous progression target was discretized into three distinct severity levels:
- **Low Severity**: Target value < 100
- **Moderate Severity**: Target value between 100 and 200 (inclusive)
- **High Severity**: Target value > 200

### 3.2 Data Splitting
The dataset was partitioned into training and testing subsets to evaluate generalization performance.
- **Split Ratio:** 80% Training, 20% Testing.
- **Implementation:** The `train_test_split` function from the `sklearn.model_selection` library was employed, ensuring a reproducible random state.

## 4. Model Implementation
The classification model was implemented using the `DecisionTreeClassifier`. Key steps included:
1. **Feature Scaling:** All 10 physiological features were pre-scaled (mean-centered) as per the dataset's standard.
2. **Hyperparameter Selection:** A `max_depth` of 5 was selected to prevent overfitting while allowing the tree to capture complex decision boundaries.
3. **Training:** The model was fitted to the training data using the Gini impurity criterion to measure the quality of splits.

## 5. Results & Evaluation
### 5.1 Classification Accuracy
The model was evaluated on the unseen testing set (89 samples).
- **Classification Accuracy:** **52.81%**

### 5.2 Success Statement
The model achieved an accuracy significantly higher than random chance (which would be ~33.3% for three classes). While the accuracy of 52.81% indicates a "Moderate" success for this specific discretized problem, it demonstrates that baseline physiological markers can indeed predict severity categories. For higher clinical precision, further refinement using ensemble methods or refined thresholds is recommended.

## 6. Conclusion & Appendix
The project successfully demonstrated the application of Decision Tree logic to clinical severity classification. 

### Appendix: Python Implementation Code
```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
diab = load_diabetes()
X, y = diab.data, diab.target

# 2. Data Discretization (Severity Labels)
y_class = np.zeros_like(y)
y_class[y < 100] = 0           # Low
y_class[(y >= 100) & (y <= 200)] = 1 # Moderate
y_class[y > 200] = 2           # High

# 3. Data Splitting (80/20 ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.20, random_state=42
)

# 4. Model Implementation
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 5. Results Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Testing Set Accuracy: {accuracy:.4f}")
```
