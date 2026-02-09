import sys

print(sys.executable)
import pandas as pd
import numpy as np 
import csv


def filter_invalid_records(data):
    """
    Remove orders that were not grabbed by a courier
    """
    return data[data["is_courier_grabbed"] == 1].copy()


def remove_missing_or_zero_timestamps(data):
    """
    Remove records with missing or zero values in key timestamp fields
    """
    required_cols = ["arrive_time", "estimate_arrived_time"]
    # Drop NaNs
    data = data.dropna(subset=required_cols)
    # Remove zero timestamps
    for col in required_cols:
        data = data[data[col] != 0]
    return data.copy()


def check_timestamp_consistency(data, max_hours=5):
    """
    Ensures estimate_arrived_time is after platform_order_time
    and within a reasonable range
    """
    data["platform_order_time"] = pd.to_datetime(data["platform_order_time"])
    data["estimate_arrived_time"] = pd.to_datetime(data["estimate_arrived_time"])

    diff_hours = (
        data["estimate_arrived_time"] - data["platform_order_time"]
    ).dt.total_seconds() / 3600
    data = data[(diff_hours > 0) & (diff_hours <= max_hours)]
    return data.copy()

# valgfitt - fjerner rader med inkonsistente faktiske tider
def check_actual_time_consistency(data):
    """
    Remove records where actual arrival time is before dispatch
    """
    data["arrive_time"] = pd.to_datetime(data["arrive_time"])
    data["dispatch_time"] = pd.to_datetime(data["dispatch_time"])
    data = data[data["arrive_time"] >= data["dispatch_time"]]
    return data.copy()
