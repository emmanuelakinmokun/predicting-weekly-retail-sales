import pandas as pd

numeric_columns = ["store", "holiday_flag", "weekly_sales", "temperature",
                "fuel_price", "cpi", "unemployment"]

def load_walmart(path):
    '''  Read the dataset and return a clean DataFrame sorted by date.
    Steps: lowercase column names, parse the day-month-year dates,
    make sure numeric columns are numbers, sort oldest to newest.
    '''
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
    return df.sort_values('date').reset_index(drop=True)





