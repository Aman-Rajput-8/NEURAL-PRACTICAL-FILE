import pandas as pd

# Experiment 2: Pandas Series and DataFrame Operations
# Aim: To demonstrate core operations on Pandas Series and DataFrames.

print("--- 1. Pandas Series Operations ---")
# Create a basic Pandas Series with custom index letters
series_data = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'], name="Scores")
print("Original Series:\n", series_data)

# Test out basic Series functionalities
print("\nMaximum Value in Series:", series_data.max())
print("Mean (Average) of Series:", series_data.mean())
print("Filtering Series (values > 25):\n", series_data[series_data > 25])

print("\n--- 2. Pandas DataFrame Operations ---")
# Create a DataFrame using a standard python dictionary
data_dict = {
    "Student": ["Alice", "Bob", "Charlie", "David"],
    "Age": [24, 27, 22, 32],
    "Grade": [88.5, 92.0, 79.5, 95.0]
}
df = pd.DataFrame(data_dict)
print("Original DataFrame:\n", df)

# Demonstrate basic DataFrame inspection
print("\nDataFrame Dimensions (Shape):", df.shape)
print("\nFetching the top 2 rows using .head():\n", df.head(2))

# Extract a single column (this returns a Series object automatically)
extracted_series = df['Grade']
print("\nExtracted 'Grade' Column (Type: Series):\n", extracted_series)

# Sort the entire DataFrame based on the Age column
df_sorted = df.sort_values(by="Age", ascending=False)
print("\nDataFrame sorted by Age (Desc):\n", df_sorted)
