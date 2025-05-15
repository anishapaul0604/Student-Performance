import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load and preprocess data
data = pd.read_csv('StudentsPerformance.csv')

# Calculate average score and create pass/fail label (using 60 as passing threshold)
data['average_score'] = (data['math score'] + data['reading score'] + data['writing score']) / 3
data['pass_fail'] = (data['average_score'] >= 60).astype(int)

# Select features for prediction
features = ['gender', 'race/ethnicity', 'parental level of education', 
           'lunch', 'test preparation course']

# Create dummy variables for categorical features
X = pd.get_dummies(data[features])
y = data['pass_fail']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model with better parameters
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)

# Fit the model
rf_model.fit(X_train, y_train)

# Evaluate model
y_pred = rf_model.predict(X_test)
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and feature columns
model_bundle = {
    'rf_model': rf_model,
    'columns': X.columns.tolist(),
    'feature_names': features
}
joblib.dump(model_bundle, 'student_pass_predictor.pkl')

# Test prediction on sample data
sample = X_test.iloc[0:1]
print("\nSample Prediction:")
print("Input features:", sample.iloc[0])
print("Predicted:", "Pass" if rf_model.predict(sample)[0] == 1 else "Fail")
print("Actual:", "Pass" if y_test.iloc[0] == 1 else "Fail")
