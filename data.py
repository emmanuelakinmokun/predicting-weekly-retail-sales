import pandas as pd

# Load and view
df = pd.read_csv('Walmart.csv')

# Convert the Date column to a proper datetime format
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

# Sort the entire dataframe by date and reset the index
df = df.sort_values(by ='Date', ascending=True).reset_index(drop=True)

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# Check for duplicated rows
duplicates = df[df.duplicated(keep=False)]

# Check for missing values
print(df.isnull().sum())

# Check for negative numbers
df_clean = df.copy()
cols_to_convert = df_clean.columns.drop('date')
df_clean[cols_to_convert] = df_clean[cols_to_convert].apply(pd.to_numeric)
print((df_clean.select_dtypes(include='number')< 0).sum())

# Check for invalid holiday
invalid_holiday = df_clean[~df_clean['holiday_flag'].isin([0,1])]
print(len(invalid_holiday))

# Check for invalid fuel prices
invalid_fuel = df_clean[df_clean['fuel_price'] <= 0]
print(len(invalid_fuel))