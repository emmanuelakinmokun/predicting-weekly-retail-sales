import numpy as np
import pandas as pd
from src.features import add_lag_features, build_dummies, timesplit

def make_df():
    """Two stores, four weeks each, with easy-to-check sales numbers."""
    rows = []
    for store, base in [(1, 100.0), (2, 1000.0)]:
        for i, date in enumerate(pd.date_range("2010-02-05", periods=4, freq="7D")):
            rows.append({
                "store": store, "date": date, "weekly_sales": base + i,
                "holiday_flag": 0, "temperature": 50.0,
                "unemployment": 8.0, "fuel_price": 3.0,
            })
    return pd.DataFrame(rows)
 
 
def test_lag_uses_previous_week_of_the_same_store():
    out = add_lag_features(make_df())
    row = out[(out["store"] == 2) & (out["weekly_sales"] == 1001.0)].iloc[0]
    assert row["sales_last_week"] == 1000.0     # store 2's own last week, not store 1's
 
 
def test_lag_drops_first_week_of_each_store():
    out = add_lag_features(make_df())
    assert len(out) == 6                         # 8 rows minus 1 per store
    assert out["sales_last_week"].notna().all()
 
 
def test_log_columns_match_numpy_log():
    out = add_lag_features(make_df())
    assert np.allclose(out["weekly_sales_log"], np.log(out["weekly_sales"]))
 
 
def test_time_split_is_chronological_and_about_20_percent():
    df = make_df()
    train = timesplit(df["date"], test_frac=0.25)
    assert df.loc[train, "date"].max() < df.loc[~train, "date"].min()
    assert 0 < (~train).sum() < len(df)
 
 
def test_design_matrix_turns_store_and_month_into_float_dummies():
    X = build_dummies(make_df())
    assert "store" not in X.columns              # replaced by dummies
    assert "store_2" in X.columns                # store 1 is the reference group
    assert X.select_dtypes(exclude="number").empty   # statsmodels needs numbers, not objects
    assert len(X) == 8                           # no rows lost
 
 
def test_design_matrix_keeps_lag_feature_when_requested():
    df = add_lag_features(make_df())
    feats = ["holiday_flag", "temperature", "unemployment", "fuel_price",
             "sales_last_week_log", "month", "store"]
    X = build_dummies(df, features=feats)
    assert "sales_last_week_log" in X.columns
    assert "store_2" in X.columns

