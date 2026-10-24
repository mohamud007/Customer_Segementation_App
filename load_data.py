
import pandas as pd

# Load the dataset
df = pd.read_csv("data/Mall_Customers.csv")

# 1. Display the first five rows
print("\n--- First Five Customers ---")
print(df.head())

# 2. Check rows and columns
print("\n--- Dataset Shape ---")
print(df.shape)

# 3. Check column names and data types
print("\n--- Dataset Information ---")
df.info()

# 4. Check missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 5. Display basic statistics
print("\n--- Statistical Summary ---")
print(df.describe())
