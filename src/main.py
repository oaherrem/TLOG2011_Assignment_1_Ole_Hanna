import os
print(os.getcwd())

import pandas as pd
from pathlib import Path
from preprocessing import (
    filter_invalid_records,
    remove_missing_or_zero_timestamps,
    check_timestamp_consistency
)

# Finn datasett
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "all_waybill_info_meituan_0322.csv"

# Les data
data = pd.read_csv(DATA_PATH, encoding="utf-8")

# Data cleaning
data = filter_invalid_records(data)
data = remove_missing_or_zero_timestamps(data)
data = check_timestamp_consistency(data)

print(data.shape)
print(data.head())
