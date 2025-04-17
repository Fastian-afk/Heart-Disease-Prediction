# ❤️ Heart Disease Prediction using Machine Learning

This project presents a machine learning-based solution for predicting the presence of heart disease using a clinical dataset. The aim is to support medical professionals by offering fast, interpretable, and scalable diagnostic assistance.

---

## 📌 Project Overview

Heart disease is a leading cause of mortality globally. Early detection is critical for effective treatment, but conventional diagnostic processes are often time-consuming and require multiple medical tests.  
This project demonstrates how machine learning can aid in faster and more accurate diagnosis using tabular patient data.

---

## 🚀 Objectives

- Predict the presence or absence of heart disease
- Compare performance across multiple classification algorithms
- Perform hyperparameter tuning for optimal model accuracy
- Handle class imbalance using SMOTE
- Evaluate using metrics like accuracy, confusion matrix, ROC-AUC, and Precision-Recall
- Visualize model performance with relevant graphs

---

## 📂 Dataset

- Source: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets) *(replace with actual link)*
- Features include: `age`, `sex`, `cp (chest pain)`, `trestbps (resting blood pressure)`, `chol (serum cholesterol)`, `thalach (max heart rate)`, and others
- Target column: `HeartDisease` (1 = disease, 0 = no disease)

---

## 🧠 Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- XGBoost Classifier

Each model was trained on the same dataset and compared using accuracy metrics and evaluation plots.

---

## 🔧 Techniques & Tools

| Technique | Description |
|----------|-------------|
| **Train/Test Split** | Used an 80/20 split for evaluation |
| **SMOTE** | Synthetic Minority Over-sampling to balance dataset |
| **GridSearchCV** | Hyperparameter tuning (for Random Forest) |
| **Cross-Validation** | 5-Fold cross-validation to validate model stability |
| **ROC Curve & PR Curve** | Visual evaluation of classification thresholds |

---

## 📊 Evaluation Metrics

- Accuracy
- Confusion Matrix
- ROC AUC Score
- Precision-Recall Curve
- Cross-Validation Accuracy

All metrics were computed for each model to assess their reliability and diagnostic potential.

---

## 📉 Visualizations

- **Confusion Matrix**: Understand true vs. false predictions
- **ROC Curve**: Compare sensitivity vs. specificity
- **Precision-Recall Curve**: Evaluate classifier performance on imbalanced data
- *(Optional)* **Time Series Trends**: If `Year` column is present in data

---

## 💻 Technologies Used

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

---

## 📈 Sample Results

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | 0.85     |
| SVM                 | 0.86     |
| Random Forest       | 0.91 ✅  |
| KNN                 | 0.88     |
| XGBoost             | 0.90     |

Random Forest and XGBoost showed the best results and were used for further evaluation with SMOTE and ROC analysis.

---

## 🧪 How to Run

1. Clone the repository  
   `git clone https://github.com/your-username/heart-disease-prediction.git`

2. Install dependencies  
   `pip install -r requirements.txt`

3. Run the notebook  
   `jupyter notebook notebooks/Heart_Disease_Analysis.ipynb`

---

## 📌 Future Work

- Integrate with web dashboards using Streamlit or Flask
- Use SHAP or LIME for better model interpretability
- Explore deep learning models on larger datasets

---

## 🤝 Let's Connect

If you're working on healthcare + AI projects, I'd love to connect and collaborate!  
Feel free to star the repo ⭐ and share your feedback.

---

## 📎 License

This project is open-source and available under the MIT License.
