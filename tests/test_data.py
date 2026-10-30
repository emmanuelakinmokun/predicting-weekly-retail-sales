import pandas as pd
from src.data import load_walmart

def test_load_walmart_cleans_columns_date_and_order(tmp_path):
    csv = tmp_path/ 'w.csv'
    csv.write_text(
        "Store,Date,Weekly_Sales,Holiday_Flag,Temperature,Fuel_Price,CPI,Unemployment\n"
        "1,12-02-2010,100.0,0,40.0,2.5,210.0,8.1\n"
        "1,05-02-2010,90.0,1,38.0,2.5,210.0,8.1\n"
    )
    df = load_walmart(csv)
    assert "weekly_sales" in df.columns                 # lowercase names
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].is_monotonic_increasing           # oldest first
    assert df.loc[0, "date"] == pd.Timestamp("2010-02-05")  # day-month-year parsed correctly

