import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load the raw dataset (uses ';' instead of ',' to separate columns)
df = pd.read_csv('bank-additional-full.csv', sep=';')

# Step 2: Convert target column 'y' from text ('yes'/'no') to numbers (1/0)
X = df.drop(columns=['y'])
y = df['y'].map({'no': 0, 'yes': 1})

# Step 3: Split into 80% train / 20% test before imputation and encoding
# Replacement values and encoding categories are learned from training only
# to prevent test data from influencing preprocessing (data leakage).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train = X_train.copy()
X_test = X_test.copy()

# Step 4: Impute 'unknown' values in simple columns with the most common value (mode)
# Calculate the mode from training only, then apply it to both sets.
simple_cols = ['job', 'marital', 'housing', 'loan']

for col in simple_cols:
    most_common = X_train.loc[
        X_train[col] != 'unknown', col
    ].mode()[0]

    X_train[col] = X_train[col].replace('unknown', most_common)
    X_test[col] = X_test[col].replace('unknown', most_common)

# Step 5: Impute 'unknown' education based on the most common education for each job
# Learn job-specific education modes from training only.
# Use the overall training education mode if a job has no known education values.
education_fallback = X_train.loc[
    X_train['education'] != 'unknown', 'education'
].mode()[0]

education_by_job = {}

for job_title in X_train['job'].unique():
    known_edu = X_train.loc[
        (X_train['job'] == job_title) &
        (X_train['education'] != 'unknown'),
        'education'
    ]

    education_by_job[job_title] = (
        known_edu.mode()[0]
        if not known_edu.empty
        else education_fallback
    )

# Apply the same training-derived replacements to both sets.
# The fallback also handles jobs that appear only in the test set.
for dataset in [X_train, X_test]:
    unknown_education = dataset['education'] == 'unknown'

    replacements = (
        dataset.loc[unknown_education, 'job']
        .map(education_by_job)
        .fillna(education_fallback)
    )

    dataset.loc[unknown_education, 'education'] = replacements

# Note on 'default' column:
# 'default' is NOT imputed. Its 8,597 'unknown' values in the raw dataset
# are intentionally kept as a separate category because unknown credit
# history may carry predictive signal; this should be evaluated during modeling.
# In Step 6 below, 'default' gets One-Hot Encoded into two numeric columns:
# 'default_unknown' and 'default_yes' (with 'no' as the 0/0 baseline),
# provided all three categories appear in training.

# Step 6: Convert all categorical text columns into numeric 0/1 columns (One-Hot Encoding)
# Fit the encoder on training only, then apply the same encoding to both sets.
categorical_cols = [
    'job', 'marital', 'education', 'default', 'housing',
    'loan', 'contact', 'month', 'day_of_week', 'poutcome'
]

numeric_cols = [
    col for col in X_train.columns
    if col not in categorical_cols
]

encoder = OneHotEncoder(
    drop='first',
    handle_unknown='ignore',
    sparse_output=False,
    dtype=int
)

encoder.fit(X_train[categorical_cols])


def encode_features(data):
    encoded = encoder.transform(data[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=data.index
    )

    return pd.concat([data[numeric_cols], encoded_df], axis=1)


X_train_encoded = encode_features(X_train)
X_test_encoded = encode_features(X_test)

# Feature matrix dimensions (without target column 'y'):
# Expected if training contains the same categories as the original script:
# X_train_encoded: 32,950 rows, 48 columns
# X_test_encoded:   8,238 rows, 48 columns

# Check that both sets have matching columns, aligned labels, and no missing features.
assert X_train_encoded.columns.equals(X_test_encoded.columns)
assert X_train_encoded.index.equals(y_train.index)
assert X_test_encoded.index.equals(y_test.index)
assert not X_train_encoded.isna().any().any()
assert not X_test_encoded.isna().any().any()

# Step 7: Combine features + target and save clean CSV files
train_df = pd.concat([X_train_encoded, y_train], axis=1)  # Expected: 32,950 rows, 49 columns (48 features + 'y')
test_df = pd.concat([X_test_encoded, y_test], axis=1)     # Expected:  8,238 rows, 49 columns (48 features + 'y')

train_df.to_csv('bank_train_clean.csv', index=False)
test_df.to_csv('bank_test_clean.csv', index=False)

print(f"Saved train_df: {train_df.shape}")  # Expected: (32950, 49)
print(f"Saved test_df:  {test_df.shape}")   # Expected: (8238, 49)import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load the raw dataset (uses ';' instead of ',' to separate columns)
df = pd.read_csv('bank-additional-full.csv', sep=';')

# Step 2: Convert target column 'y' from text ('yes'/'no') to numbers (1/0)
X = df.drop(columns=['y'])
y = df['y'].map({'no': 0, 'yes': 1})

# Step 3: Split into 80% train / 20% test before imputation and encoding
# Replacement values and encoding categories are learned from training only
# to prevent test data from influencing preprocessing (data leakage).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_train = X_train.copy()
X_test = X_test.copy()

# Step 4: Impute 'unknown' values in simple columns with the most common value (mode)
# Calculate the mode from training only, then apply it to both sets.
simple_cols = ['job', 'marital', 'housing', 'loan']

for col in simple_cols:
    most_common = X_train.loc[
        X_train[col] != 'unknown', col
    ].mode()[0]

    X_train[col] = X_train[col].replace('unknown', most_common)
    X_test[col] = X_test[col].replace('unknown', most_common)

# Step 5: Impute 'unknown' education based on the most common education for each job
# Learn job-specific education modes from training only.
# Use the overall training education mode if a job has no known education values.
education_fallback = X_train.loc[
    X_train['education'] != 'unknown', 'education'
].mode()[0]

education_by_job = {}

for job_title in X_train['job'].unique():
    known_edu = X_train.loc[
        (X_train['job'] == job_title) &
        (X_train['education'] != 'unknown'),
        'education'
    ]

    education_by_job[job_title] = (
        known_edu.mode()[0]
        if not known_edu.empty
        else education_fallback
    )

# Apply the same training-derived replacements to both sets.
# The fallback also handles jobs that appear only in the test set.
for dataset in [X_train, X_test]:
    unknown_education = dataset['education'] == 'unknown'

    replacements = (
        dataset.loc[unknown_education, 'job']
        .map(education_by_job)
        .fillna(education_fallback)
    )

    dataset.loc[unknown_education, 'education'] = replacements

# Note on 'default' column:
# 'default' is NOT imputed. Its 8,597 'unknown' values in the raw dataset
# are intentionally kept as a separate category because unknown credit
# history may carry predictive signal; this should be evaluated during modeling.
# In Step 6 below, 'default' gets One-Hot Encoded into two numeric columns:
# 'default_unknown' and 'default_yes' (with 'no' as the 0/0 baseline),
# provided all three categories appear in training.

# Step 6: Convert all categorical text columns into numeric 0/1 columns (One-Hot Encoding)
# Fit the encoder on training only, then apply the same encoding to both sets.
categorical_cols = [
    'job', 'marital', 'education', 'default', 'housing',
    'loan', 'contact', 'month', 'day_of_week', 'poutcome'
]

numeric_cols = [
    col for col in X_train.columns
    if col not in categorical_cols
]

encoder = OneHotEncoder(
    drop='first',
    handle_unknown='ignore',
    sparse_output=False,
    dtype=int
)

encoder.fit(X_train[categorical_cols])


def encode_features(data):
    encoded = encoder.transform(data[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=data.index
    )

    return pd.concat([data[numeric_cols], encoded_df], axis=1)


X_train_encoded = encode_features(X_train)
X_test_encoded = encode_features(X_test)

# Feature matrix dimensions (without target column 'y'):
# Expected if training contains the same categories as the original script:
# X_train_encoded: 32,950 rows, 48 columns
# X_test_encoded:   8,238 rows, 48 columns

# Check that both sets have matching columns, aligned labels, and no missing features.
assert X_train_encoded.columns.equals(X_test_encoded.columns)
assert X_train_encoded.index.equals(y_train.index)
assert X_test_encoded.index.equals(y_test.index)
assert not X_train_encoded.isna().any().any()
assert not X_test_encoded.isna().any().any()

# Step 7: Combine features + target and save clean CSV files
train_df = pd.concat([X_train_encoded, y_train], axis=1)  # Expected: 32,950 rows, 49 columns (48 features + 'y')
test_df = pd.concat([X_test_encoded, y_test], axis=1)     # Expected:  8,238 rows, 49 columns (48 features + 'y')

train_df.to_csv('bank_train_clean.csv', index=False)
test_df.to_csv('bank_test_clean.csv', index=False)

print(f"Saved train_df: {train_df.shape}")  # Expected: (32950, 49)
print(f"Saved test_df:  {test_df.shape}")   # Expected: (8238, 49)