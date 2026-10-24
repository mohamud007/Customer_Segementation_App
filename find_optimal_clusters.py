
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load customer data
df = pd.read_csv("data/Mall_Customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Calculate inertia for different cluster counts
inertia_values = []

for k in range(1, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X)
    inertia_values.append(model.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia_values, marker="o")
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.tight_layout()
plt.savefig("outputs/elbow_method.png")
plt.show()

print("Elbow Method graph saved successfully!")
