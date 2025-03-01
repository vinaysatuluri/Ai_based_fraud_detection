# Step 1: Dataset Loading
import pandas as pd
import numpy as np
import os 
from sklearn.model_selection import train_test_split

# Load dataset
dataset_path = os.path.join(os.path.dirname(__file__), "../dataset/fraud_detection_dataset.csv")
df = pd.read_csv(dataset_path)
print("Dataset Loaded Successfully! ✅")
print("Dataset Preview:\n", df.head())

# Step 2: Data Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Drop completely missing column (Check_Number)
df = df.drop(columns=["Check_Number"], errors="ignore")

# Encode categorical columns using Label Encoding
categorical_columns = df.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert boolean values to integers
df = df.astype({col: "int" for col in df.select_dtypes(include=["bool"]).columns})

# Define features & target
X = df.drop(columns=["Suspicious_Activity_Flag"])
y = df["Suspicious_Activity_Flag"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nData Preprocessing Completed! ✅")
print("Processed Data Sample:\n", X_train.head())

# Step 3: Model Comparison
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    results.append([name, accuracy, precision, recall, f1, roc_auc])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
print("\nModel Comparison Completed! ✅")
print(results_df)

# Step 4: Handling Imbalance with SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("\nSMOTE Applied! ✅")
print("Resampled Class Distribution:")
print(pd.Series(y_train_resampled).value_counts())

# Step 5: Feature Engineering
from sklearn.feature_selection import SelectFromModel
feature_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
feature_selector.fit(X_train_resampled, y_train_resampled)
X_train_selected = feature_selector.transform(X_train_resampled)
X_test_selected = feature_selector.transform(X_test)
selected_features = X.columns[feature_selector.get_support()]
print("\nFeature Engineering Completed! ✅")
print("Selected Features:\n", selected_features)

# Step 6: Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
param_grid_rf = {"n_estimators": [100, 200], "max_depth": [10, 20]}
best_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='f1')
best_rf.fit(X_train_selected, y_train_resampled)
best_rf = best_rf.best_estimator_

param_grid_xgb = {"n_estimators": [100, 200], "max_depth": [3, 6]}
best_xgb = GridSearchCV(XGBClassifier(eval_metric='logloss'), param_grid_xgb, cv=3, scoring='f1')
best_xgb.fit(X_train_selected, y_train_resampled)
best_xgb = best_xgb.best_estimator_

print("\nHyperparameter Tuning Completed! ✅")
print("Best Random Forest Params:", best_rf.get_params())
print("Best XGBoost Params:", best_xgb.get_params())

# Step 7: Training and Evaluation
def evaluate_model(name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n{name} Performance:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

y_pred_rf = best_rf.predict(X_test_selected)
y_pred_xgb = best_xgb.predict(X_test_selected)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)

# Step 8: Weighted Voting Classifier
from sklearn.ensemble import VotingClassifier
weighted_voting_clf = VotingClassifier(estimators=[("RF", best_rf), ("XGB", best_xgb)], voting="soft", weights=[0.6, 0.4])
weighted_voting_clf.fit(X_train_selected, y_train_resampled)
print("\nEnsemble Model Training Completed! ✅")

# Step 9: Saving the Model
import joblib
model_save_path = os.path.join(os.path.dirname(__file__), "../models/final_fraud_detection_model.pkl")
joblib.dump(weighted_voting_clf, model_save_path, compress=3)


# Step 10: Evaluating on Test Data
final_model = joblib.load(model_save_path)
final_pred = final_model.predict(X_test_selected)
evaluate_model("Final Ensemble Model", y_test, final_pred)
