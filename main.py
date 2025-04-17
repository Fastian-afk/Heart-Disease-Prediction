import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the dataset
df = pd.read_csv(r"C:\Users\Hp\Downloads\archive\Heart_Prediction_Quantum_Dataset.csv")
# 🔁 Replace with the actual CSV file path

# Step 2: Basic data inspection
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Preprocess the data
df.dropna(inplace=True)  # Drop missing values

# Encode categorical data
if 'Gender' in df.columns:
    df['Gender'] = LabelEncoder().fit_transform(df['Gender'])  # Male=1, Female=0

# Identify target column (update this if your column name is different)
target_column = 'HeartDisease'  # 🛑 Make sure this is correct

# Features and labels
X = df.drop(target_column, axis=1)
y = df[target_column]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 4: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 7: Feature importance
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importances')
plt.tight_layout()
plt.show()

# Step 8: Predict on new data (example)
# Update with real values when needed
new_data = np.array([[65, 1, 120, 230, 150, 0.82]])  # Example row
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)

print("\nPrediction for new data:", "Heart Disease" if prediction[0] == 1 else "No Heart Disease")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split the data into features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predicting heart disease for a new person
# Replace the values below with actual input: [Age, Gender, BloodPressure, Cholesterol, HeartRate, QuantumPatternFeature]
new_data = [[60, 1, 140, 240, 150, 0.76]]  # example input
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("🔴 This person is likely to have heart disease.")
else:
    print("🟢 This person is unlikely to have heart disease.")

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# 1. Histogram of key features
df.hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
plt.suptitle("Distribution of Features", fontsize=16)
plt.tight_layout()
plt.show()

# 2. Countplot of Heart Disease
sns.countplot(data=df, x='HeartDisease', palette='Set2')
plt.title("Heart Disease Cases (0 = No, 1 = Yes)")
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# 4. Feature Importance Barplot
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 6. Simulated Prediction Trend (Fake Data)
future_years = list(range(2025, 2031))
simulated_predictions = np.random.randint(30, 70, size=len(future_years))

plt.figure(figsize=(8, 5))
sns.lineplot(x=future_years, y=simulated_predictions, marker='o')
plt.title("Simulated Heart Disease Prediction Trend (2025–2030)")
plt.ylabel("Predicted Cases (%)")
plt.xlabel("Year")
plt.grid(True)
plt.tight_layout()
plt.show()

# Additional Imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# 1. Model Comparison with Accuracy Scores
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy}")

# 2. Hyperparameter Tuning using GridSearchCV (RandomForest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters (RandomForest):", grid_search.best_params_)

# 3. K-Fold Cross Validation
cv_scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)
print(f"Average 5-Fold CV Accuracy: {cv_scores.mean()}")

# 4. SMOTE for Imbalanced Class Distribution (if applicable)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 5. Model Training (on resampled data)
model = RandomForestClassifier()
model.fit(X_train_resampled, y_train_resampled)

# 6. Confusion Matrix, ROC Curve, and Classification Report
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# 7. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

# 8. Accuracy Over Years (If Year Data is Present)
if 'Year' in df.columns:
    yearly_data = df.groupby('Year')['HeartDisease'].sum()
    plt.figure(figsize=(10, 6))
    yearly_data.plot(kind='line', marker='o', linestyle='-', color='b')
    plt.title("Heart Disease Occurrence Over Years")
    plt.xlabel('Year')
    plt.ylabel('Number of Heart Disease Cases')
    plt.grid(True)
    plt.show()

# End of enhancements