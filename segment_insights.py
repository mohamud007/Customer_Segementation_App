
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load customer data
df = pd.read_csv("data/Mall_Customers.csv")

# Select the features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)

# Calculate segment statistics
summary = df.groupby("Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2)

# Generate simple business insights
def describe_segment(row):
    income = row["Average_Income"]
    spending = row["Average_Spending"]

    if income >= 70 and spending >= 60:
        return "High income, high spending"
    elif income >= 70 and spending < 40:
        return "High income, low spending"
    elif income < 40 and spending >= 60:
        return "Low income, high spending"
    elif income < 40 and spending < 40:
        return "Low income, low spending"
    else:
        return "Moderate or mixed spending"

summary["Segment_Description"] = summary.apply(
    describe_segment, axis=1
)

# Display the report
print("--- Customer Segment Insights ---")
print(summary.to_string())

# Save the report
summary.to_csv("outputs/segment_insights.csv")

print("\nSegment insights saved successfully!")
