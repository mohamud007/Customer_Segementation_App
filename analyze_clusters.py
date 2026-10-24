
import pandas as pd
from sklearn.cluster import KMeans

# Load customer data
df = pd.read_csv("data/Mall_Customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Train K-Means
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# Calculate the average values for each cluster
cluster_summary = df.groupby("Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending_Score=("Spending Score (1-100)", "mean")
).round(2)

# Display the results
print("--- Customer Segment Analysis ---")
print(cluster_summary)

# Save the analysis
cluster_summary.to_csv("outputs/cluster_summary.csv")

# Save customer records with their cluster assignments
df.to_csv("outputs/segmented_customers.csv", index=False)

print("\nAnalysis saved successfully!")
