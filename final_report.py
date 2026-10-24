
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select features
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means
model = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)

# Generate summary
summary = df.groupby("Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2)

# Save customer-level results and summary
df.to_csv("outputs/final_segmented_customers.csv", index=False)
summary.to_csv("outputs/final_segment_summary.csv")

# Create final visualization
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Mall Customer Segmentation - K-Means")
plt.xlabel("Annual Income (thousands)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Customer Cluster")
plt.tight_layout()
plt.savefig("outputs/final_customer_segmentation.png", dpi=300)
plt.show()

# Print final report
print("\n--- FINAL CUSTOMER SEGMENT REPORT ---")
print(summary.to_string())

print("\nTotal customers analyzed:", len(df))
print("Number of clusters:", df["Cluster"].nunique())
print("Final report generated successfully!")
