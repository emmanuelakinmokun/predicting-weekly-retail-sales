import pandas as pd

# Load and view
df = pd.read_csv('Walmart.csv')
print(df.head())

# Convert the Date column to a proper dattime format
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

# Sort the entire dataframe by date and reset the index
df = df.sort_values(by ='Date', ascending=True).reset_index(drop=True)

print(df.head())

