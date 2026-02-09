import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def select_direct_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Selects direct input features from the dataset without modification
    """

    direct_feature_cols = [
        # Order characteristics
        "is_prebook",
        "is_weekend",

        # Location identifiers
        "poi_id",
        "da_id",

        # Geographic coordinates (scaled integers)
        "sender_lat",
        "sender_lng",
        "recipient_lat",
        "recipient_lng",
    ]

    return data[direct_feature_cols].copy()

# 2.4.2 Engineered Features (Calculated from Raw Data)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def to_unix_seconds(series: pd.Series) -> pd.Series:
    """
    Ensures timestamps are in Unix seconds.
    Handles both int and datetime64 inputs.
    """
    if np.issubdtype(series.dtype, np.datetime64):
        return series.astype("int64") // 10**9
    return series


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # --- Ensure Unix timestamps ---
    data["platform_order_time_unix"] = to_unix_seconds(
        data["platform_order_time"]
    )
    data["estimate_arrived_time_unix"] = to_unix_seconds(
        data["estimate_arrived_time"]
    )

    # --- Time features from platform_order_time ---
    data["order_datetime"] = pd.to_datetime(
        data["platform_order_time_unix"], unit="s"
    )

    data["order_hour"] = data["order_datetime"].dt.hour
    data["order_dayofweek"] = data["order_datetime"].dt.dayofweek
    data["order_is_weekend"] = data["order_dayofweek"].isin([5, 6]).astype(int)

    # Peak hours: lunch (11–14) and dinner (17–20)
    data["order_is_peak"] = data["order_hour"].isin(
        [11, 12, 13, 14, 17, 18, 19, 20]
    ).astype(int)

    # --- Coordinate conversion ---
    data["sender_lat_deg"] = data["sender_lat"] / 1e6
    data["sender_lng_deg"] = data["sender_lng"] / 1e6
    data["recipient_lat_deg"] = data["recipient_lat"] / 1e6
    data["recipient_lng_deg"] = data["recipient_lng"] / 1e6

    # --- Merchant–customer distance ---
    data["merchant_customer_distance_km"] = data.apply(
        lambda row: haversine(
            row["sender_lat_deg"],
            row["sender_lng_deg"],
            row["recipient_lat_deg"],
            row["recipient_lng_deg"],
        ),
        axis=1,
    )

    # --- Estimated meal preparation time (minutes) ---
    data["estimated_prep_time_minutes"] = (
        data["estimate_meal_prepare_time"]
        - data["platform_order_time_unix"]
    ) / 60

    data.loc[
        (data["estimated_prep_time_minutes"] < 0)
        | (data["estimated_prep_time_minutes"] > 120),
        "estimated_prep_time_minutes",
    ] = np.nan

    # --- Estimated delivery duration (minutes) ---
    data["estimated_delivery_duration_minutes"] = (
        data["estimate_arrived_time_unix"]
        - data["platform_order_time_unix"]
    ) / 60

    # --- Cleanup ---
    data.drop(
        columns=[
            "order_datetime",
            "platform_order_time_unix",
            "estimate_arrived_time_unix",
            "sender_lat_deg",
            "sender_lng_deg",
            "recipient_lat_deg",
            "recipient_lng_deg",
        ],
        inplace=True,
        errors="ignore",
    )

    return data
