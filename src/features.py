import pandas as pd

#2.4.1 Direct features (from dataset)
def select_direct_features(data):
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


# 2.4.2 Engineerd Features (Calculated from raw data)