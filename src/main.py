import pandas as pd
import numpy as np
from pathlib import Path

from preprocessing import (
    filter_invalid_records,
    remove_missing_or_zero_timestamps,
    check_timestamp_consistency,
)
from features import select_direct_features, add_engineered_features
from train import train_logistic_regression, train_random_forest, train_gradient_boosting
from evaluate import evaluate_classification

#Thresold tuning
PREDICTION_THRESHOLD = 0.60


# Load data
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "all_waybill_info_meituan_0322.csv"

data = pd.read_csv(DATA_PATH, encoding="utf-8")


# Data cleaning (2.3)
data = filter_invalid_records(data)
data = remove_missing_or_zero_timestamps(data)
data = check_timestamp_consistency(data)
#print(data.shape)
#print(data.head())


# Feature selection (2.4.1)
X_direct = select_direct_features(data)
#print("Feature matrix shape:", X_direct.shape)
#print(X_direct.head())


# Feature engineering (2.4.2)
X_engineered = add_engineered_features(data)
#print("Feature matrix with engineered features shape:", X_engineered.shape)
#print(X_engineered.head())


# Keep only engineered features
engineered_cols = [c for c in X_engineered.columns if c not in X_direct.columns]
X_engineered = X_engineered[engineered_cols]

# Final feature dataframe
X = X_direct.join(X_engineered)
#print(X.head())

# Target variable (2.5)
data["estimate_arrived_time"] = pd.to_numeric(data["estimate_arrived_time"])
data["arrive_time"] = pd.to_numeric(data["arrive_time"])

# tolerance
tolerance_min = 8 #minutes
#tolerance_seconds = tolerance_min

data["late_delivery"] = (
    data["arrive_time"] > data["estimate_arrived_time"] + tolerance_min
).astype(int)

y_class = data["late_delivery"]

# Feature matrix for model
feature_cols = [
    "is_prebook",
    "is_weekend",
    "poi_id",
    "da_id",
    "sender_lat",
    "sender_lng",
    "recipient_lat",
    "recipient_lng",
    "order_hour",
    "order_dayofweek",
    "order_is_weekend",
    "order_is_peak",
    "merchant_customer_distance_km",
    "estimated_delivery_duration_minutes",
]

X_model = X[feature_cols]


# Remove rows with missing values
valid_idx = X_model.notna().all(axis=1) & y_class.notna()

X_model = X_model.loc[valid_idx]
y_class = y_class.loc[valid_idx]
platform_order_time = data.loc[valid_idx,"platform_order_time"]

# Time-based train/test split (2.5)
split_df = X_model.assign(
    platform_order_time = platform_order_time,
    y_class=y_class
).sort_values("platform_order_time")

split_idx = int(0.8 * len(split_df))


train_df = split_df.iloc[:split_idx]
test_df = split_df.iloc[split_idx:]

X_train = train_df[feature_cols]
y_train = train_df["y_class"]

X_test = test_df[feature_cols]
y_test = test_df["y_class"]

# Train model (2.5)
model = train_logistic_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)
gb_model = train_gradient_boosting(X_train, y_train)

print("Train class distribution:")
print(y_train.value_counts(normalize=True))

print("Test class distribution:")
print(y_test.value_counts(normalize=True))

# Evaluate model (2.5)
lr_metrics = evaluate_classification(model, X_test, y_test)

print("Model evaluation results:")
print(f"Accuracy:  {lr_metrics['accuracy']:.3f}")
print(f"Precision: {lr_metrics['precision']:.3f}")
print(f"Recall:    {lr_metrics['recall']:.3f}")
print("Confusion Matrix:")
print(lr_metrics["confusion_matrix"])


rf_metrics = evaluate_classification(rf_model, X_test, y_test)

print("\nRandom Forest results:")
print(f"Accuracy:  {rf_metrics['accuracy']:.3f}")
print(f"Precision: {rf_metrics['precision']:.3f}")
print(f"Recall:    {rf_metrics['recall']:.3f}")
print("Confusion Matrix:")
print(rf_metrics["confusion_matrix"])


gb_metrics = evaluate_classification(gb_model, X_test, y_test)

print("\nGradient Boosting results:")
print(f"Accuracy:  {gb_metrics['accuracy']:.3f}")
print(f"Precision: {gb_metrics['precision']:.3f}")
print(f"Recall:    {gb_metrics['recall']:.3f}")
print("Confusion Matrix:")
print(gb_metrics["confusion_matrix"])

# Visualize results. look in results-folder for saved images
from visualize import plot_confusion_matrix, plot_model_comparison, plot_roc_curves
import os

os.makedirs("results", exist_ok=True)

plot_confusion_matrix(
    lr_metrics["confusion_matrix"],
    "Logistic Regression – Confusion Matrix",
    "results/confusion_matrix_lr.png"
)

plot_confusion_matrix(
    rf_metrics["confusion_matrix"],
    "Random Forest – Confusion Matrix",
    "results/confusion_matrix_rf.png"
)

plot_confusion_matrix(
    gb_metrics["confusion_matrix"],
    "Gradient Boosting – Confusion Matrix",
    "results/confusion_matrix_gb.png"
)

plot_model_comparison(
    {
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "Gradient Boosting": gb_metrics,
    },
    "results/model_comparison.png"
)

models = {
    "Logistic Regression": model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
}

plot_roc_curves(models, X_test, y_test)
