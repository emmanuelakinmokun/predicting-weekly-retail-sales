import os
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

# Create a report dir
os.makedirs('reports')

# Build data-quaity summary metric
dq_metrics = []

for col in df_clean.columns:
    total_rows = len(df)
    missing_count = df_clean[col].isnull().sum()
    dtype = str(df_clean[col].dtype) # except datetime

# Check if column is numeric to calculate stats and outliers
if pd.api.types.is_numeric_dtype(df[col]):
    min_val = df_clean[col].min()
    max_val = df_clean[col].max()

    # Outlier detection using IQR
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

else:
    min_val = df[col].min() if col == 'date' else 'N/A'
    max_val = df[col].max() if col == 'date' else 'N/A'
    outlier_count = 'N/A'

dq_metrics.append({
    'Column Name': col,
      'Data Type': dtype,
      'Total Rows': total_rows,
      'Missing Count': missing_count,
      'Min Value': min_val,
      'Max Value': max_val,
      'Outliers Flagged (IQR)': outlier_count,
})

dq_df =pd.DataFrame(dq_metrics)

# Save to reports
output_path = 'reports/data_quality_report.csv'
dq_df.to_csv(output_path, index=False)

 # Split dataset for training
def time_split(df, test_frac=0.2):
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()