# Turning the clean data into model inputs (X) and a time-based split.

import numpy as np
import pandas as pd

base_features = ["holiday_flag", "temperature", "unemployment", "fuel_price", "month", "store"]
categorical  = ["month", "store"]

def add_lag_features(df):
    #  Add last week's sales (per store) and log versions of sales.
    #  The first week of each store has no 'last week', so those rows are dropped.
    #  Returns a new DataFrame; the input is not changed.

    out = df.sort_values(['store', 'date']).copy()
    out["sales_last_week"] = out.groupby("store")["weekly_sales"].shift(1)
    out = out.dropna(subset=["sales_last_week"])
    out["sales_last_week_log"] = np.log(out["sales_last_week"])
    out["weekly_sales_log"] = np.log(out["weekly_sales"])
    return out

def build_dummies(df, features=base_features):
    # return X: the chosen columns, with month and store turned into 0/1 dummies.
    # drop_first=True leaves out one month and one store as the reference group.
    # The index of df is kept, so X lines up with df and y.

    data = df.copy()
    data["month"] = data["date"].dt.month
    cats = [c for c in categorical if c in features]
    return pd.get_dummies(data[features], columns=cats, drop_first=True, dtype=float)

def timesplit(dates, test_frac=0.2):
    # Training rows: everything before the cut

    weeks = np.sort(dates.unique())
    cut = weeks[int(len(weeks) * (1 - test_frac))]
    return dates < cut


 
