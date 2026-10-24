
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("--- Original Feature Averages ---")
print(X.mean().round(2))

print("\n--- Scaled Feature Averages ---")
print(pd.DataFrame(X_scaled, columns=X.columns).mean().round(4))

# Train K-Means on scaled features
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Visualize the original income and spending values
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=80
)

plt.title("Customer Segmentation with Scaled Features")
plt.tight_layout()
plt.savefig("outputs/scaled_customer_segments.png")
plt.show()

# Save the segmented data
df.to_csv("outputs/scaled_segmented_customers.csv", index=False)

print("\nScaled model completed successfully!")
