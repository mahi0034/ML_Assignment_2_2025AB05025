# ML Assignment 2 - Classification Models with Streamlit Deployment

## 🎯 Problem Statement

This project implements and compares six different machine learning classification models on the Credit Card Fraud Detection dataset. The goal is to predict fraudulent transactions based on 30 anonymized features, demonstrating the end-to-end ML workflow including model training, evaluation, web application development, and cloud deployment.

---

## 📊 Dataset Description

**Dataset Name:** Credit Card Fraud Detection  
**Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
**Type:** Binary Classification  

### Dataset Characteristics:
- **Total Instances:** 284,807
- **Total Features:** 30 (V1-V28 from PCA transformation, Time, Amount)
- **Target Variable:** Class
- **Class Distribution:** 
  - Class 0 (Normal): 284,315 (99.83%)
  - Class 1 (Fraud): 492 (0.17%)

### Features Overview:
This dataset contains transactions made by European credit cardholders in September 2013. It presents transactions that occurred over two days, with 492 frauds out of 284,807 transactions.

- **Features V1-V28:** Principal components obtained using PCA transformation to protect user identities and sensitive features
- **Time:** Seconds elapsed between this transaction and the first transaction in the dataset
- **Amount:** Transaction amount
- **Class:** Binary target variable (1 = fraud, 0 = normal)

### Data Preprocessing:
- Train-Test Split: 80%-20% (227,845 training samples, 56,962 test samples)
- Feature Scaling: StandardScaler applied to all features
- Handling Missing Values: No missing values in dataset
- Class Balancing: Highly imbalanced dataset (0.17% fraud) - models evaluated with appropriate metrics (MCC, AUC)

---

## 🤖 Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| **Logistic Regression** | 0.9991 | 0.9605 | 0.9991 | 0.9991 | 0.9991 | 0.7228 |
| **Decision Tree** | 0.9991 | 0.8722 | 0.9991 | 0.9991 | 0.9991 | 0.7483 |
| **K-Nearest Neighbor** | 0.9995 | 0.9437 | 0.9995 | 0.9995 | 0.9995 | 0.8603 |
| **Naive Bayes** | 0.9764 | 0.9632 | 0.9981 | 0.9764 | 0.9865 | 0.2195 |
| **Random Forest (Ensemble)** | 0.9996 | 0.9630 | 0.9996 | 0.9996 | 0.9996 | 0.8763 |
| **XGBoost (Ensemble)** | 0.9994 | 0.9390 | 0.9994 | 0.9994 | 0.9994 | 0.8303 |

---

## 📝 Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Achieved 99.91% accuracy on the highly imbalanced credit card fraud dataset, indicating that the linear decision boundary is already very effective in this feature space. The F1 Score of 0.9991 shows a strong balance between precision and recall at the aggregate level, and the AUC of 0.9605 reflects good discriminative ability between fraudulent and non-fraudulent transactions. Fast training and the interpretability of coefficients make it a strong baseline model. |
| **Decision Tree** | Reached 99.91% accuracy with an MCC of 0.7483, which indicates a strong correlation between predictions and true labels despite the severe class imbalance. The model can capture non-linear relationships and interactions between features, but its lower AUC of 0.8722 compared to other models suggests it is somewhat less robust in ranking fraud vs non-fraud instances. The tree structure and feature importance still make it useful for interpretability and exploratory analysis. |
| **K-Nearest Neighbor** | Instance-based learning achieved 99.95% accuracy with an MCC of 0.8603 and an AUC of 0.9437, indicating very strong overall performance. The model benefits from the standardized feature space but remains sensitive to the choice of k (k=5 used) and distance metric, and can be computationally expensive at prediction time on large datasets like this. It performs well here but may not scale as efficiently as the ensemble methods in production settings. |
| **Naive Bayes** | As a probabilistic classifier, Naive Bayes obtained 97.64% accuracy with very high precision (0.9981) and an F1 Score of 0.9865, showing good aggregate performance. However, the MCC of 0.2195 is much lower than for the other models, suggesting that it is less effective at correctly identifying the minority fraud class in this highly imbalanced setting. Its main advantages are extremely fast training and prediction, making it suitable as a lightweight baseline rather than the best-performing model. |
| **Random Forest (Ensemble)** | The Random Forest ensemble delivered the best overall performance with 99.96% accuracy, an AUC of 0.9630, and the highest MCC of 0.8763 among all models, indicating excellent discrimination and reliability on the fraud class. By aggregating many decision trees through bagging, it reduces variance and overfitting compared to a single tree, and provides stable predictions on this imbalanced dataset. Its feature importance scores also help identify which transformed variables contribute most to fraud detection. |
| **XGBoost (Ensemble)** | XGBoost achieved 99.94% accuracy with an F1 Score of 0.9994, an MCC of 0.8303, and an AUC of 0.9390, demonstrating very strong performance close to Random Forest. Although literature often reports XGBoost as a top performer on fraud detection tasks, in these experiments Random Forest slightly outperforms it in both AUC and MCC. Nevertheless, XGBoost benefits from gradient boosting and regularization, making it a powerful choice for imbalanced problems and a strong alternative to Random Forest on this dataset. |

---

## 🚀 Streamlit Web Application

### Features Implemented:
✅ **Dataset Upload**: CSV file upload with validation  
✅ **Model Selection**: Dropdown menu for 6 classification models  
✅ **Evaluation Metrics Display**: Accuracy, AUC, Precision, Recall, F1, MCC  
✅ **Confusion Matrix**: Visual heatmap with true/predicted labels  
✅ **Classification Report**: Detailed per-class performance metrics  
✅ **Interactive Dashboard**: Real-time predictions with downloadable results  

### Live Application:
🔗 **Streamlit App:** https://2025ab05025.streamlit.app/

---

## 📁 Repository Structure

```
project-folder/
│
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── creditcard.csv
├── test.csv
├── bits_lab_execution.png
├── model/                      # Saved trained models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── k-nearest_neighbor.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl             # Feature scaler
    ├── results.pkl            # Evaluation results
    └── model_all.py           # Training script for all 6 models


```

---

## 🛠️ Installation & Usage

### Local Setup:

```bash
# Clone the repository
git clone https://github.com/mahi0034/ML_Assignment_2_2025AB05025.git
cd ML_Assignment_2_2025AB05025

# Install dependencies
pip install -r requirements.txt

# Train models
python model_training.py

# Run Streamlit app
streamlit run app.py
```

### Cloud Deployment:
Application deployed on **Streamlit Community Cloud**  
Access via: https://2025ab05025.streamlit.app/

---

## 📊 Evaluation Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **AUC (Area Under ROC Curve)**: Model's ability to discriminate between classes
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient)**: Balanced measure for imbalanced datasets


---

## 👨‍💻 Author

**Name:** Mahender Singh
**Student ID:** 2025AB05025
**Email:** 2025ab05025@wilp.bits-pilani.ac.in  

---

