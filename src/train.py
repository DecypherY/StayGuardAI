# ============================================================
# STAYGUARD AI — MODEL TRAINING AND EVALUATION
# ============================================================

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed_bookings.csv"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

data = pd.read_csv(DATA_PATH)

print("Original processed dataset shape:", data.shape)


# ============================================================
# 3. FEATURE SELECTION
# ============================================================

FEATURES = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "adr",
]

TARGET = "last_minute_cancel"

X = data[FEATURES].copy()
y = data[TARGET].copy()

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(include=["number"]).columns.tolist()

categorical_features = [
    column for column in FEATURES if column not in numerical_features
]

print("\nNumerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))


# ============================================================
# 5. SPLIT DATA INTO TRAINING, VALIDATION AND TESTING
# ============================================================

# First: Keep 30% completely untouched for final testing.

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Second: Divide the remaining 70% into training and validation.
# Validation represents 20% of the remaining 70%.

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.20, random_state=42, stratify=y_train_val
)

print("\nDataset split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)


# ============================================================
# 6. NUMERICAL PREPROCESSING
# ============================================================

numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


# ============================================================
# 7. CATEGORICAL PREPROCESSING
# ============================================================

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# ============================================================
# 8. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_transformer, numerical_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)


# ============================================================
# 9. CREATE RANDOM FOREST MODEL
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# 10. CREATE COMPLETE MACHINE LEARNING PIPELINE
# ============================================================

model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", random_forest)])


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ============================================================
# 12. VALIDATION AND THRESHOLD TUNING
# ============================================================

# Get cancellation probabilities on validation data.

val_probabilities = model.predict_proba(X_val)[:, 1]

# Calculate precision and recall at different thresholds.

precision, recall, thresholds = precision_recall_curve(y_val, val_probabilities)

# Calculate F1 for every available threshold.

f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)

# Select the threshold with the highest validation F1.

best_index = f1_scores.argmax()
best_threshold = float(thresholds[best_index])

print("\nBest validation threshold:", round(best_threshold, 4))
print("Best validation F1-score:", round(f1_scores[best_index], 4))


# ============================================================
# 13. FINAL TEST PREDICTIONS
# ============================================================

# Test probabilities are generated only after threshold selection.

test_probabilities = model.predict_proba(X_test)[:, 1]

# Apply the threshold selected using validation data.

test_predictions = (test_probabilities >= best_threshold).astype(int)


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

print("\n======================================")
print("FINAL TEST RESULTS")
print("======================================")

print("\nClassification Report:")

report = classification_report(y_test, test_predictions, zero_division=0)

print(report)


print("\nConfusion Matrix:")

matrix = confusion_matrix(y_test, test_predictions)

print(matrix)


roc_auc = roc_auc_score(y_test, test_probabilities)

avg_precision = average_precision_score(y_test, test_probabilities)

test_f1 = f1_score(y_test, test_predictions, zero_division=0)

print("\nROC-AUC:", round(roc_auc, 4))
print("Average Precision:", round(avg_precision, 4))
print("Test F1-score:", round(test_f1, 4))


# ============================================================
# 15. SAVE TRAINED MODEL
# ============================================================

MODEL_PATH = MODEL_DIR / "stayguard_model.joblib"

joblib.dump(
    {"model": model, "threshold": best_threshold, "features": FEATURES}, MODEL_PATH
)

print("\nModel saved to:", MODEL_PATH)


# ============================================================
# 16. SAVE EVALUATION RESULTS
# ============================================================

metrics = {
    "threshold": best_threshold,
    "validation_f1": float(f1_scores[best_index]),
    "test_f1": float(test_f1),
    "roc_auc": float(roc_auc),
    "average_precision": float(avg_precision),
    "confusion_matrix": matrix.tolist(),
}

METRICS_PATH = OUTPUT_DIR / "model_metrics.json"

with open(METRICS_PATH, "w") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved to:", METRICS_PATH)

print("\n======================================")
print("STAYGUARD AI TRAINING COMPLETED")
print("======================================")
