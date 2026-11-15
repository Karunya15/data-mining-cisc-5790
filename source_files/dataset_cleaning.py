import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the raw dataset (uses ';' instead of ',' to separate columns)
df = pd.read_csv('bank-additional-full.csv', sep=';')

# Step 2: Impute 'unknown' values in simple columns with the most common value (mode)
simple_cols = ['job', 'marital', 'housing', 'loan']
for col in simple_cols:
    most_common = df[df[col] != 'unknown'][col].mode()[0]
    df[col] = df[col].replace('unknown', most_common)

# Step 3: Impute 'unknown' education based on the most common education for each job
for job_title in df['job'].unique():
    known_edu = df[(df['job'] == job_title) & (df['education'] != 'unknown')]['education']
    most_common_edu = known_edu.mode()[0]
    df.loc[(df['job'] == job_title) & (df['education'] == 'unknown'), 'education'] = most_common_edu

# Note on 'default' column:
# 'default' is NOT imputed. Its 8,597 'unknown' values are intentionally kept as a
# separate category because missing credit history carries predictive signal.
# In Step 4 below, 'default' gets One-Hot Encoded into two numeric columns:
# 'default_unknown' and 'default_yes' (with 'no' as the 0/0 baseline).

# Step 4: Convert all categorical text columns into numeric 0/1 columns (One-Hot Encoding)
categorical_cols = [
    'job', 'marital', 'education', 'default', 'housing',
    'loan', 'contact', 'month', 'day_of_week', 'poutcome'
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# Step 5: Convert target column 'y' from text ('yes'/'no') to numbers (1/0)
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# Step 6: Split into features (X) and target (y), then 80% train / 20% test
X = df.drop(columns=['y'])
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Feature matrix dimensions (without target column 'y'):
# X_train: 32,950 rows, 48 columns
# X_test:   8,238 rows, 48 columns

# Step 7: Combine features + target and save clean CSV files
train_df = pd.concat([X_train, y_train], axis=1)  # 32,950 rows, 49 columns (48 features + 'y')
test_df = pd.concat([X_test, y_test], axis=1)     #  8,238 rows, 49 columns (48 features + 'y')

train_df.to_csv('bank_train_clean.csv', index=False)
test_df.to_csv('bank_test_clean.csv', index=False)

print(f"Saved train_df: {train_df.shape}")  # (32950, 49)
print(f"Saved test_df:  {test_df.shape}")   # (8238, 49)