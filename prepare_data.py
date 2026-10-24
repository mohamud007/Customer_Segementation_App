
import pandas as pd

# Load the customer dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select the features used for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Display the selected data
print("--- Selected Features ---")
print(X.head())

# Check the number of customers and features
print("\nShape of selected data:")
print(X.shape)

# Check for missing values
print("\nMissing values:")
print(X.isnull().sum())
