import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# -------------------- Load dataset --------------------
# Use the extended dataset we just generated
csv_path = 'fitness_dataset_extended.csv'   # change if you saved with another name
if not os.path.exists(csv_path):
    # fallback to the old name if needed
    csv_path = 'fitness_dataset.csv'

df = pd.read_csv(csv_path)

# -------------------- Prepare features --------------------
# Numeric columns expected in the extended dataset
numeric_cols = [c for c in [
    'Age', 'Height_cm', 'Weight_kg', 'BMI',
    'WorkoutsPerWeek', 'WalksPerWeek'
] if c in df.columns]

# Categorical columns to one-hot encode (keep only those that exist)
categorical_cols = [c for c in [
    'Gender', 'FitnessGoal', 'Level', 'HealthCondition', 'ActivityLevel'
] if c in df.columns]

# One-Hot Encoding (adds *_<category> columns, removes originals)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# -------------------- Label encode targets --------------------
# Robustly find target columns
if 'WorkoutType' not in df_encoded.columns or 'DietType' not in df_encoded.columns:
    raise ValueError("Dataset must contain 'WorkoutType' and 'DietType' columns.")

le_workout = LabelEncoder()
df_encoded['WorkoutType_enc'] = le_workout.fit_transform(df_encoded['WorkoutType'])

le_diet = LabelEncoder()
df_encoded['DietType_enc'] = le_diet.fit_transform(df_encoded['DietType'])

# -------------------- Build X / y --------------------
# Features = all columns except the raw target strings and encoded targets
target_cols = ['WorkoutType', 'DietType', 'WorkoutType_enc', 'DietType_enc']
feature_cols = [c for c in df_encoded.columns if c not in target_cols]

X = df_encoded[feature_cols].astype(float)  # RF handles unscaled features fine
y_workout = df_encoded['WorkoutType_enc']
y_diet = df_encoded['DietType_enc']

# Save the feature column order for inference time
joblib.dump(feature_cols, 'feature_columns.pkl')

# -------------------- Train / test split (same split for both tasks) --------------------
X_train, X_test, y_train_workout, y_test_workout, y_train_diet, y_test_diet = train_test_split(
    X, y_workout, y_diet, test_size=0.2, random_state=42
)

# -------------------- Train models --------------------
rf_workout = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
).fit(X_train, y_train_workout)

rf_diet = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
).fit(X_train, y_train_diet)

# -------------------- Evaluate --------------------
pred_w = rf_workout.predict(X_test)
pred_d = rf_diet.predict(X_test)

print("WorkoutType Accuracy:", accuracy_score(y_test_workout, pred_w))
print("DietType Accuracy:", accuracy_score(y_test_diet, pred_d))

# -------------------- Save models & encoders --------------------
joblib.dump(rf_workout, 'rf_workout.pkl')
joblib.dump(rf_diet, 'rf_diet.pkl')
joblib.dump(le_workout, 'le_workout.pkl')
joblib.dump(le_diet, 'le_diet.pkl')

print("✅ Models trained & saved!")