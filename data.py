import pandas as pd

# Load and view
df = pd.read_csv('Walmart.csv')
print(df.head())

# Convert the Date column to a proper dattime format
df['Date'] = pd.to_datetime(df['Date'])

