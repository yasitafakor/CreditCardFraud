### Project Overview

This project involves a comprehensive analysis and classification of credit card transactions to detect fraudulent activity. The dataset used is highly imbalanced, with a vast majority of transactions classified as non-fraudulent. To tackle this, various machine learning techniques were applied, including both supervised (Logistic Regression) and unsupervised (KMeans Clustering) learning methods.

The process began with exploratory data analysis, where the dataset was inspected for missing values, distribution of classes, and relevant features. The data was preprocessed by normalizing the 'Amount' feature and removing irrelevant columns like 'Time.' 

The next phase involved building a Logistic Regression model. Initially, the model was trained on the original imbalanced dataset, followed by applying oversampling (using SMOTE) and undersampling techniques to balance the dataset. The performance of each approach was evaluated using metrics like accuracy, recall, precision, F1 score, and AUC.

To further enhance the analysis, KMeans clustering was implemented to explore how an unsupervised method might group the transactions. The results were analyzed by comparing the clustering outcomes against the true labels, generating confusion matrices and evaluating the same metrics.

Finally, the project includes a comparative analysis of the performance between Logistic Regression and KMeans, illustrated through various plots, and an exploration of feature correlations within the identified clusters.

This project showcases proficiency in data preprocessing, handling class imbalances, applying multiple machine learning techniques, and evaluating model performance comprehensively in the context of fraud detection.

**Data Loading and Exploration:**
   - Loaded the credit card transaction dataset and created a copy for analysis.
   - Conducted initial analysis using `.describe()` to understand data statistics.

**Handling Missing Values:**
   - Checked for missing values using `.isna()`, confirming the dataset is complete.

**Data Visualization:**
   - Visualized class distribution with `sns.countplot`, showing 284,315 non-fraudulent (class 0) and 492 fraudulent transactions (class 1).
   - Created scatter plots to observe class separation.

**Data Preparation:**
   - Removed the 'Time' column as it is irrelevant.
   - Split the dataset into features (X) and target (y), where X contains all columns except 'Class', and y is the 'Class' column indicating fraud.

![first](https://github.com/user-attachments/assets/f9717c6b-8806-4557-bc11-471ee6ccbe32)


## Logistic Regression Implementation

In this section, I implemented and evaluated a Logistic Regression model to classify credit card transactions as fraudulent or non-fraudulent. The goal was to assess the model's performance on predicting fraudulent transactions, given the imbalanced nature of the dataset.

### Data Preparation
1. **Train-Test Split:**
   - The dataset was split into training and testing sets using an 80-20 split ratio (`test_size=0.2`). The random state was set to 42 for reproducibility.

2. **Feature Scaling:**
   - The 'Amount' feature was standardized using `StandardScaler` to ensure all features have similar scales.

### Model Training
1. **Logistic Regression:**
   - A Logistic Regression model was trained on the scaled training data with a maximum of 200 iterations and a random state of 0 for reproducibility.

### Model Evaluation
1. **Predictions:**
   - Predictions were made on the test set using the trained model.

### Results
- **Recall:** 0.5816
- **F1 Score:** 0.6951
- **Accuracy:** 99.9122%

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 1.00   | 1.00     | 56,864  |
| 1     | 0.86      | 0.58   | 0.70     | 98      |

**Confusion Matrix Heatmap:**

![11](https://github.com/user-attachments/assets/d7518735-29b8-4ffc-8e94-51ef0cb058cf)

## Handling Class Imbalance

Given the significant class imbalance in the credit card fraud dataset, where the majority class (non-fraudulent transactions) vastly outnumbers the minority class (fraudulent transactions), we applied both oversampling and undersampling techniques to improve the model's ability to detect fraud.

### 1. Oversampling with SMOTE

**Evaluation Metrics:**
- **Recall:** 0.9184
- **F1 Score:** 0.1091
- **Accuracy:** 97.4193%

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 0.97   | 0.99     | 56,864  |
| 1     | 0.06      | 0.92   | 0.11     | 98      |

![22](https://github.com/user-attachments/assets/c2997517-717f-4b65-8305-6bf5fe5c0a8e)


**Observations:**
- The recall for the minority class improved significantly to 91.84%, indicating better detection of fraudulent transactions.
- However, precision and F1-score for the minority class remained low, suggesting a higher number of false positives.

### 2. Undersampling with RandomUnderSampler

**Evaluation Metrics:**
- **Recall:** 0.9286
- **F1 Score:** 0.1110
- **Accuracy:** 97.4422%

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 0.97   | 0.99     | 56,864  |
| 1     | 0.06      | 0.93   | 0.11     | 98      |

![33](https://github.com/user-attachments/assets/b84afcc3-d4cd-48bf-97ed-c7321046d2f4)

**Observations:**
- The recall for the minority class further improved to 92.86%, indicating a slight enhancement in detecting fraudulent transactions.
- Similar to the oversampling approach, the F1-score remained low, highlighting the trade-off between recall and precision when handling class imbalance.

### Conclusion

- Both oversampling and undersampling **improved** the model's ability to detect fraudulent transactions, as evidenced by the increased recall. However, these methods also led to a reduction in precision and F1-score, indicating the challenges of achieving high performance across all metrics when dealing with highly imbalanced datasets.
- The choice between oversampling and undersampling depends on the specific requirements of the **application**, particularly the need for ** recall versus precision**.

### KMeans Clustering Implementation

In addition to Logistic Regression, KMeans clustering was employed to explore an unsupervised approach to classifying the data into two clusters: Fraud and Non-Fraud.

- KMeans was used with `n_clusters=2` to group the dataset into two clusters.
- The distribution of class 0 (Non-Fraud) and class 1 (Fraud) across the two clusters was analyzed.
- A confusion matrix was constructed to evaluate the performance of KMeans in identifying the two classes.

**Cluster 1 as Fraud and Cluster 2 as Non-Fraud:**

```python
Accuracy Score:  0.53
Recall Score:  0.27
Precision Score:  0.0
F1 Score:  0.0
AUC Score:  0.4
```

**Cluster 2 as Fraud and Cluster 1 as Non-Fraud:**

```python
Accuracy Score:  0.47
Recall Score:  0.73
Precision Score:  0.0
F1 Score:  0.0
AUC Score:  0.6
```

**Final Results:**
- **Max Recall:** 0.73
- **Max AUC:** 0.6
- **Max Accuracy:** 0.53

### Comparative Analysis: Logistic Regression vs. KMeans

To better understand the differences in performance between Logistic Regression and KMeans, the following plots were generated:

#### Accuracy Comparison

![Acc](https://github.com/user-attachments/assets/89922060-24ff-4e58-9d86-7a0f5968b129)

#### AUC Score Comparison

![AUC](https://github.com/user-attachments/assets/400ce05e-3334-448b-b241-ea28b722acad)

#### Recall Score Comparison

![Recall](https://github.com/user-attachments/assets/8fa8bbef-95cb-446b-9621-a5240fa439f8)

### Correlation Matrices for Clusters

Finally, correlation matrices were plotted for each cluster to understand the relationships between the features.

#### Correlation Matrix for Cluster 1
![hm1](https://github.com/user-attachments/assets/966b8721-a611-4e34-a913-8105199542e9)


#### Correlation Matrix for Cluster 2
![hm2](https://github.com/user-attachments/assets/809db49c-0ac0-4eab-b85e-6ce2340fb15b)

