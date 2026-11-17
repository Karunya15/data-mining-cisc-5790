import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC #SMOTENC handles both numeric and categorical values 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load the raw dataset (uses ';' instead of ',' to separate columns)
df = pd.read_csv('bank-additional-full.csv', sep=';')

# Step 2: Split into features (X) and target (y), then 80% train / 20% test
#Note: We moved the split to before imputation and one-hot encoding to avoid data leakage
X = df.drop(columns=['y'])
y = df['y'].map({'no': 0, 'yes': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Step 3: Impute 'unknown' values in simple columns with the most common value (mode)
simple_cols = ['job', 'marital', 'housing', 'loan']
for col in simple_cols:
    most_common = X_train.loc[X_train[col] != 'unknown', col].mode()[0]
    X_train[col] = X_train[col].replace('unknown', most_common)
    X_test[col] = X_test[col].replace('unknown', most_common)

# Step 4: Impute 'unknown' education based on the most common education for each job

# Fallback if a job has no known education values in training
education_fallback = X_train.loc[X_train['education'] != 'unknown', 'education'].mode()[0]

for job_title in X_train['job'].unique():
    known_edu = X_train.loc[(X_train['job'] == job_title) & (X_train['education'] != 'unknown'),'education']
    most_common_edu = (known_edu.mode()[0] if not known_edu.empty else education_fallback)

    for dataset in [X_train, X_test]: 
        dataset.loc[(dataset['job'] == job_title) & (dataset['education'] == 'unknown'), 'education'] = most_common_edu

# Step 5: Identify/separate categorical and numeric features
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

# Step 6: Scale numeric features using training statistics
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Step 7: Oversample the minority class in training only
smote = SMOTENC(categorical_features=categorical_cols, sampling_strategy=1.0, random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Step 8: Learn category columns from the original training set through one-hot encoding
encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

encoder.fit(X_train_scaled[categorical_cols])

# Apply the same encoding to data
def encode_features(data):
    encoded = encoder.transform(data[categorical_cols])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=data.index)
    return pd.concat([data[numeric_cols], encoded_df], axis=1)

# Apply the same encoding to all three datasets so they have matching columns
X_train_encoded = encode_features(X_train_scaled)
X_resampled_encoded = encode_features(X_train_resampled)
X_test_encoded = encode_features(X_test_scaled)

# Step 9: Attach labels by row position and save
def save_dataset(features, labels, filename):
    output = features.reset_index(drop=True).copy()
    output['y'] = labels.to_numpy()
    output.to_csv(filename, index=False)
    print(f"Saved {filename}: {output.shape}")

# assert X_train_encoded.columns.equals(X_resampled_encoded.columns)
# assert X_train_encoded.columns.equals(X_test_encoded.columns)

# assert len(X_train_encoded) == len(y_train)
# assert len(X_resampled_encoded) == len(y_train_resampled)
# assert len(X_test_encoded) == len(y_test)

save_dataset(X_train_encoded, y_train,'bank_train_baseline_scaled.csv')
save_dataset(X_resampled_encoded, y_train_resampled, 'bank_train_smotenc_scaled.csv')
save_dataset(X_test_encoded, y_test, 'bank_test_scaled.csv')

# Note on 'default' column:
# 'default' is NOT imputed. Its 8,597 'unknown' values are intentionally kept as a
# separate category because missing credit history carries predictive signal.

#Examine class discrepency (if present) to determine class imbalance
print(f"Training class counts: {y_train.value_counts()}") # (0- 29238) (1- 3712)
#Normalize the data and get its percentage (*100)
print(f"Training class percentages: {y_train.value_counts(normalize=True) * 100}") # (0- 88.734) (1- 11.266)

print("Class counts after SMOTENC:")
print(y_train_resampled.value_counts()) #25,526 synthetic minority-class examples added - Total of 58,476 training rows
