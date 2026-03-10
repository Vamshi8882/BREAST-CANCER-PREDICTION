# ============================================
# Breast Cancer Prediction using ML Algorithms
# ============================================

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Dataset
data = pd.read_csv("DataSet Breast Cancer/data.csv")

print("Dataset Shape:", data.shape)
print("\nFirst 5 Rows:\n")
print(data.head())

# 3. Dataset Information
print("\nDataset Info:\n")
print(data.info())

# 4. Check Missing Values
print("\nMissing Values:\n")
print(data.isnull().sum())

# 5. Drop Unnecessary Columns
if 'id' in data.columns:
    data.drop('id', axis=1, inplace=True)

if 'Unnamed: 32' in data.columns:
    data.drop('Unnamed: 32', axis=1, inplace=True)

# 6. Encode Target Variable
# M → 1 (Malignant), B → 0 (Benign)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

print("\nTarget Distribution:\n")
print(data['diagnosis'].value_counts())

# 7. Feature and Target Split
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 9. Feature Scaling (for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# 10. Logistic Regression Model
# --------------------------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

lr_accuracy = accuracy_score(y_test, lr_pred) * 100

print("\n===== Logistic Regression =====")
print(f"Accuracy: {lr_accuracy:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, lr_pred))

# --------------------------------------------
# 11. Decision Tree Model
# --------------------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred) * 100

print("\n===== Decision Tree =====")
print(f"Accuracy: {dt_accuracy:.2f}%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, dt_pred))

# --------------------------------------------
# 12. Model Comparison
# --------------------------------------------
print("\n===== Model Comparison =====")
print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}%")
print(f"Decision Tree Accuracy: {dt_accuracy:.2f}%")

# --------------------------------------------
# 13. Confusion Matrix Visualization
# --------------------------------------------
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, lr_pred),
            annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, dt_pred),
            annot=True, fmt="d", cmap="Greens")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()