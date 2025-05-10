# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import time

# %%
X = pd.read_csv('secom/secom.data', sep=' ', header=None)
feature_names = [f'feature{i+1}' for i in range(X.shape[1])]
X.columns = feature_names
X.head()

# %%
y = pd.read_csv('secom/secom_labels.data', sep=' ', header=None)
label_columns = ['label', 'date_time']
y.columns = label_columns
y.head()

# %%
# Split the data into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y['label']
)

# Check the sizes of the resulting datasets
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# %%
# Checking Class Distributions after splitting the data
print('Original label distribution: ')
print(y['label'].value_counts(normalize=True))

print('\nTraining label distribution: ')
print(y_train['label'].value_counts(normalize=True))

print('\nTest label distribution: ')
print(y_test['label'].value_counts(normalize=True))

# %%
# Descriptive Analysis on the Test Set
X_train.sample(5, axis=1).hist(bins=30, figsize=(15, 10))
plt.suptitle('Histogram of 5 random features in the train set')
plt.show()

# %%
# Checking for duplicate rows
duplicates = X_train.duplicated()

# Count the number of duplicates
no_of_duplicates = duplicates.sum()
print(f'Number of duplicated rows in X_train: {no_of_duplicates}') # 0

# %%
# Distribution of Null Values across features
missing_percent_per_feature = (X_train.isnull().sum() / len(X_train)) * 100

plt.figure(figsize=(10,6))
sns.histplot(missing_percent_per_feature, bins=30, kde=False, color='salmon')
plt.title('Distribution of Null Value (%) Across Features')
plt.xlabel('Percent of Missing Values')
plt.ylabel('Number of Features')
plt.tight_layout()
plt.show()

features_with_high_null_values = missing_percent_per_feature[
    missing_percent_per_feature >= 44.5].index.tolist() # len is 32
# %%
# Finding features with no volatility (i.e. std dev = 0)
zero_std_dev_cols = X_train.loc[:, X_train.std() == 0].columns
zero_std_dev_cols = list(zero_std_dev_cols)
print(f"Number of zero-variance features: {len(zero_std_dev_cols)}") # 116
print("Zero-variance features:", zero_std_dev_cols)

# %%
# Combining the list of features that have a high percent of missing values and features that have 0 volatility to eventually drop
features_to_drop = features_with_high_null_values + zero_std_dev_cols
print(len(features_to_drop)) # 148

# Drop the identified features from train set for now
X_train_cleaned = X_train.drop(columns=features_to_drop)
# X_test_cleaned = X_test.drop(columns=features_to_drop)

print(f"Remaining features after drop: {X_train_cleaned.shape[1]} features.") # 442

# %%
# Impute missing values with its median and do a correlation analysis
imputer = SimpleImputer(strategy='median')

X_train_imputed = imputer.fit_transform(X_train_cleaned) 

# since the previous step converts the df to a np array, here we're converting it back to a df
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train_cleaned.columns) 

# Do the same for X_test when required
# X_test_imputed = imputer.transform(X_test_cleaned)
# X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test_cleaned.columns)

print(X_train_imputed.isnull().sum())

# %%
# Correlation matrix
correlation_matrix = X_train_imputed.corr()

# # Plot the correlation matrix as a heatmap (Takes around 7m and isn't that helpful)
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
# plt.title("Correlation Matrix")
# plt.tight_layout()
# plt.show()

# Set a threshold correlation value and extract pairs above it
THRESHOLD = 0.8

# Get upper triangle of the correlation matrix since it's symmetric
upper_triangle_corr_matrix = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

high_corr_to_drop = set()

# Iterate through upper triangle of the correlation matrix
for col in upper_triangle_corr_matrix.columns:
    correlated = upper_triangle_corr_matrix[col][
        abs(upper_triangle_corr_matrix[col]) > THRESHOLD].index.tolist()
    high_corr_to_drop.update(correlated)

# 225 when threshold is 0.8 (chose 0.8 since its used in the research paper); 198 when threshold is 0.9
print(f"Number of features to drop due to high correlation: {len(high_corr_to_drop)}")

# %%
X_train_final = X_train_imputed.drop(columns=high_corr_to_drop)
# X_test_final = X_test_imputed.drop(columns=high_corr_to_drop)

y_train_arr = y_train.drop(columns='date_time')
y_train_arr = y_train_arr.to_numpy()

# Numerical Columns
numerical_columns = X_train_final.columns.to_list()

# Numerical Preprocessing Pipeline: Impute missing values with median and scale
numerical_pipeline = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

# Combine into a Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num_pipeline', numerical_pipeline, numerical_columns)
    ]
)

# %%
# Define Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_jobs=-1),
    'SVM': SVC(class_weight='balanced')
}

# Define scoring metrics for imbalanced data (focus on class 1 = failure)
scoring = {
    'precision': make_scorer(precision_score, average='binary', pos_label=1),
    'recall': make_scorer(recall_score, average='binary', pos_label=1),
    'f1': make_scorer(f1_score, average='binary', pos_label=1)
}

# Stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run CV for each model
results = []

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', model)
    ])

    print(f"Training {model_name}...")
    start_time = time.time()
    
    y_pred = cross_val_predict(pipeline, X_train_final, y_train_arr, cv=cv)
    execution_time = round(time.time() - start_time, 2)
    
    # Compute Metrics (still targeting Fail class = 1 as positive class)
    precision = round(precision_score(y_train_arr, y_pred, pos_label=1), 4)
    recall = round(recall_score(y_train_arr, y_pred, pos_label=1), 4)
    f1 = round(f1_score(y_train_arr, y_pred, pos_label=1), 4)

    # Standard format: [[TN, FP], [FN, TP]]
    # Negative class: -1 (Pass), Positive class: 1 (Fail)
    cm_raw = confusion_matrix(y_train_arr, y_pred, labels=[-1, 1])
    tn, fp = cm_raw[0]
    fn, tp = cm_raw[1]
    cm = np.array([[tn, fp],
                   [fn, tp]])


    results.append({
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': cm,
        'Execution Time (s)': execution_time
    })

results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
print("\n📈 Model Comparison:\n")
print(results_df)