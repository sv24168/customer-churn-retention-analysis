import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# STEP 1: GENERATE MOCK TELCO DATASET
# ==========================================
print("--- Step 1: Simulating Customer Data ---")
n_samples = 5000

data = {
    'CustomerID': [f'CUST-{i:04d}' for i in range(n_samples)],
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
    'Partner': np.random.choice(['Yes', 'No'], n_samples),
    'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
    'TenureMonths': np.random.randint(1, 72, n_samples),
    'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
    'MultipleLines': np.random.choice(['No phone service', 'No', 'Yes'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.5, 0.2]),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
    'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples),
}

df = pd.DataFrame(data)

# Calculate realistic TotalCharges and simulate Churn based on logical risk factors
# Month-to-month contracts & Fiber optic users have naturally higher churn tendencies
df['TotalCharges'] = df['TenureMonths'] * df['MonthlyCharges']

churn_prob = (
    (df['Contract'] == 'Month-to-month').astype(int) * 0.4 +
    (df['InternetService'] == 'Fiber optic').astype(int) * 0.2 +
    (df['TenureMonths'] < 12).astype(int) * 0.3
)
# Normalize probabilities and add noise
churn_prob = np.clip(churn_prob + np.random.normal(0, 0.1, n_samples), 0, 1)
df['Churn'] = np.where(churn_prob > 0.5, 'Yes', 'No')

print(f"Dataset generated with shape: {df.shape}")
print(f"Churn Distribution:\n{df['Churn'].value_counts(normalize=True)}\n")


# ==========================================
# STEP 2: DATA CLEANING & TRAIN-TEST SPLIT
# ==========================================
print("--- Step 2: Cleaning Data & Splitting ---")

# Handle any unexpected spaces or missing values in TotalCharges (Common in raw churn data)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Map target label to binary integers
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Define Features and Target
X = df.drop(columns=['CustomerID', 'Churn'])
y = df['Churn']

# Stratified split ensures equal distribution of churners in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# ==========================================
# STEP 3: PREPROCESSING & PIPELINE SETUP
# ==========================================
print("--- Step 3: Structuring Preprocessing Transformers ---")

# Isolate columns by type
numeric_features = ['TenureMonths', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                        'MultipleLines', 'InternetService', 'Contract', 'PaperlessBilling', 'PaymentMethod']

# Create sub-transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# ==========================================
# STEP 4: TRAINING ENGINE WITH SMOTE
# ==========================================
print("--- Step 4: Building & Fitting Pipeline ---")

# Use imblearn.pipeline to natively chain SMOTE with Scikit-Learn
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1))
])

# Train the full model pipeline
pipeline.fit(X_train, y_train)
print("Model training successfully completed!\n")


# ==========================================
# STEP 5: EVALUATION METRICS
# ==========================================
print("--- Step 5: Model Evaluation ---")

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Print Classification Reports
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}\n")


# ==========================================
# STEP 6: BUSINESS STRATEGY & RETENTION OUTPUT
# ==========================================
print("--- Step 6: Extracting Actionable Retention List ---")

# Combine test features back with identifiers to see WHO is at risk
test_results = X_test.copy()
test_results['CustomerID'] = df.loc[X_test.index, 'CustomerID']
test_results['ActualChurn'] = y_test
test_results['ChurnRiskProbability'] = y_prob

# Sort by highest risk score
retention_action_list = test_results.sort_values(by='ChurnRiskProbability', ascending=False)

# Identify High-Risk cohorts (e.g., probability > 70%) who are Month-to-Month
high_risk_priority = retention_action_list[
    (retention_action_list['ChurnRiskProbability'] >= 0.70) & 
    (retention_action_list['Contract'] == 'Month-to-month')
]

# Calculate the financial baseline at risk
total_revenue_at_risk = high_risk_priority['MonthlyCharges'].sum()

print(f"Total customers flagged at critical risk (>70%): {len(high_risk_priority)}")
print(f"Total Monthly Recurring Revenue (MRR) at immediate risk: ${total_revenue_at_risk:,.2f}")
print("\nTop 5 At-Risk Customers to Target with Incentives:")
print(high_risk_priority[['CustomerID', 'TenureMonths', 'MonthlyCharges', 'ChurnRiskProbability']].head())


# ==========================================
# PLOT VISUALIZATIONS (Optional Run)
# ==========================================
# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()