
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select income and spending score
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Create the K-Means model
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# Train the model and assign customer groups
df["Cluster"] = kmeans.fit_predict(X)

# Display the first five customers and their groups
print("--- Customer Segments ---")
print(df[["Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].head())

# Visualize the customer groups
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=80
)

plt.title("Customer Segmentation Using K-Means")
plt.tight_layout()
plt.savefig("outputs/customer_segments.png")
plt.show()

# Count customers in each group
print("\n--- Customers in Each Cluster ---")
print(df["Cluster"].value_counts().sort_index())
