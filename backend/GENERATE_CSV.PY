import pandas as pd

# Read the original CSV file
df = pd.read_csv('emotions.csv')

# Extract the first 100 rows
df_first_100 = df.head(100)

# Save these rows to a new CSV file
df_first_100.to_csv('emotions_first_100.csv', index=False)

print("First 100 rows have been saved to 'emotions_first_100.csv'.")
