
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Give the graphs a clean style
sns.set_theme(style="whitegrid")

# Graph 1: Customer age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=15, kde=True)
plt.title("Distribution of Customer Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("outputs/customer_age_distribution.png")


# Graph 2: Annual income vs spending score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)"
)
plt.title("Annual Income vs Spending Score")
plt.tight_layout()
plt.savefig("outputs/income_vs_spending.png")
plt.show()

print("Graphs saved successfully in the outputs folder!")
