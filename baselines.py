import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve

# Load the preprocessed dataset
df = pd.read_csv("data/nhanes_preprocessed.csv")

# Drop all missing values in Has_Diabetes
df = df.dropna(subset=['Has_Diabetes'])

# Convert Has_Diabetes to an integer
df['Has_Diabetes'] = df['Has_Diabetes'].astype(int)

# Define features (X) and target (y)
X = df.drop(columns=['SEQN', 'Has_Diabetes'])  # Remove patient ID & target column
y = df['Has_Diabetes']

# Split into Training & Testing Sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a dictionary to store results
results = {}

# Model 1: Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=3,
    class_weight='balanced', random_state=42
)
rf_clf.fit(X_train, y_train)
y_rf_pred = rf_clf.predict(X_test)
y_rf_prob = rf_clf.predict_proba(X_test)[:, 1]

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_f1_scores = cross_val_score(rf_clf, X, y, cv=cv, scoring='f1')
rf_f1_mean = np.mean(rf_f1_scores)

# Store results
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_rf_pred),
    'roc_auc': roc_auc_score(y_test, y_rf_prob),
    'f1_score': rf_f1_mean
}

# Model 2: XGBoost Classifier
xgb_clf = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric='logloss'  # Removed `use_label_encoder=False`
)
xgb_clf.fit(X_train, y_train)
y_xgb_pred = xgb_clf.predict(X_test)
y_xgb_prob = xgb_clf.predict_proba(X_test)[:, 1]

# Cross-validation
xgb_f1_scores = cross_val_score(xgb_clf, X, y, cv=cv, scoring='f1')
xgb_f1_mean = np.mean(xgb_f1_scores)

# Store results
results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, y_xgb_pred),
    'roc_auc': roc_auc_score(y_test, y_xgb_prob),
    'f1_score': xgb_f1_mean
}

# Model 3: Logistic Regression
log_reg = LogisticRegression(class_weight='balanced', max_iter=2000, solver="saga")  # Increased max_iter and changed solver
log_reg.fit(X_train, y_train)
y_log_pred = log_reg.predict(X_test)
y_log_prob = log_reg.predict_proba(X_test)[:, 1]

# Cross-validation
log_f1_scores = cross_val_score(log_reg, X, y, cv=cv, scoring='f1')
log_f1_mean = np.mean(log_f1_scores)

# Store results
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_log_pred),
    'roc_auc': roc_auc_score(y_test, y_log_prob),
    'f1_score': log_f1_mean
}

# Print Model Comparison
results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df)

# Define our colors
custom_colors = ["#006778", "#6f0a19", "#e9edee"]
custom_alphas = [1.0, 0.44, 0.30]

# Plot F1 Score and ROC AUC Score Comparison for all models
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x=results_df.index, y=results_df['f1_score'], ax=axes[0], palette=custom_colors)
axes[0].set_title("F1 Score Comparison")
axes[0].set_ylim(0, 1)
axes[0].set_xlabel("Model")
axes[0].set_ylabel("F1 Score")
sns.barplot(x=results_df.index, y=results_df['roc_auc'], ax=axes[1], palette=custom_colors)
axes[1].set_title("ROC AUC Score Comparison")
axes[1].set_ylim(0, 1)
axes[1].set_xlabel("Model")
axes[1].set_ylabel("ROC AUC Score")

plt.tight_layout()
plt.show()


# Plot Feature Importance for Random Forest & XGBoost
plt.figure(figsize=(12, 6))

feature_name_map = {
    'LBXGH': 'Glycohemoglobin (HbA1c)',
    'LBXSGL': 'Serum Glucose',
    'LBXGLU': 'Fasting Glucose',
    'BMXWAIST': 'Waist Circumference',
    'Age': 'Age'
    # add as many mappings as needed
}

# Random Forest Feature Importance
rf_feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_clf.feature_importances_})
rf_feature_importance = rf_feature_importance.sort_values(by='Importance', ascending=False)

# Rename feature names using the mapping
rf_feature_importance['Feature'] = (
    rf_feature_importance['Feature']
    .map(feature_name_map)
    .fillna(rf_feature_importance['Feature'])  # Keep original name if not in map
)

plt.subplot(1, 2, 1)
ex = sns.barplot(
    y=rf_feature_importance['Feature'],
    x=rf_feature_importance['Importance'],
    hue=rf_feature_importance['Feature'],
    palette=custom_colors,
    legend=False
)
for patch, alpha in zip(ex.patches, custom_alphas):
    patch.set_alpha(alpha)
plt.xlabel("Feature Importance")
plt.ylabel("Medical Tests & Demographics")
plt.title("Random Forest Feature Importance")

# XGBoost Feature Importance
xgb_feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_clf.feature_importances_})

# Filter out features with importance less than 0.05
xgb_feature_importance = xgb_feature_importance[xgb_feature_importance['Importance'] >= 0.05]
xgb_feature_importance = xgb_feature_importance.sort_values(by='Importance', ascending=False)

# Rename feature names using the same mapping
xgb_feature_importance['Feature'] = (
    xgb_feature_importance['Feature']
    .map(feature_name_map)
    .fillna(xgb_feature_importance['Feature'])  # Keep original name if not in map
)

# Create the barplot
plt.subplot(1, 2, 2)
sns.barplot(y=xgb_feature_importance['Feature'], x=xgb_feature_importance['Importance'], hue=xgb_feature_importance['Feature'], palette="Oranges_r", legend=False)
plt.xlabel("Feature Importance")
plt.ylabel("Medical Tests & Demographics")
plt.title("XGBoost Feature Importance")
ax = sns.barplot(
    y=xgb_feature_importance['Feature'],
    x=xgb_feature_importance['Importance'],
    hue=xgb_feature_importance['Feature'],
    palette=custom_colors,
    legend=False
)
for patch, alpha in zip(ax.patches, custom_alphas):
    patch.set_alpha(alpha)

plt.xlabel("Feature Importance")
plt.ylabel("Medical Tests & Demographics")
plt.title("XGBoost Feature Importance")

plt.tight_layout()
plt.show()

# Plot ROC Curves for all models
plt.figure(figsize=(8, 6))
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_rf_prob)
plt.plot(
    fpr_rf, tpr_rf,
    label=f"Random Forest (AUC = {results['Random Forest']['roc_auc']:.2f})",
    color="#e9edee",
    alpha=0.8
)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_xgb_prob)
plt.plot(
    fpr_xgb, tpr_xgb,
    label=f"XGBoost (AUC = {results['XGBoost']['roc_auc']:.2f})",
    color="#006778",
    alpha=0.44
)

fpr_log, tpr_log, _ = roc_curve(y_test, y_log_prob)
plt.plot(
    fpr_log, tpr_log,
    label=f"Logistic Regression (AUC = {results['Logistic Regression']['roc_auc']:.2f})",
    color="#6f0a19",
    alpha=0.44
)
plt.plot([0, 1], [0, 1], linestyle="--", color='gray')  # Diagonal line for random classifier
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Different Models")
plt.legend()
plt.show()