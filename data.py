import pandas as pd

# Load and view
df = pd.read_csv('Walmart.csv')

# Convert the Date column to a proper datetime format
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

# Sort the entire dataframe by date and reset the index
df = df.sort_values(by ='Date', ascending=True).reset_index(drop=True)

# Convert column names to lowercase
df.columns = df.columns.str.lower()
print(df.columns.tolist())

