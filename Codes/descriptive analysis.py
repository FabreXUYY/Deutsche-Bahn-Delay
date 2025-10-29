import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Load the dataset
df = pd.read_excel(r"E:\3model.xlsx")

# 2️⃣ Verify that essential columns exist
required_cols = ["depart_hour_bucket", "train_category", "has_delay"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# 3️⃣ Calculate hourly on-time rates (1 - delay rate)
on_time_by_hour = (1 - df.groupby("depart_hour_bucket")["has_delay"].mean()) * 100
print("Hourly Punctuality Rate (%):")
print(on_time_by_hour.round(2))
print("\n")

# 4️⃣ Calculate on-time rates by train category
on_time_by_type = (1 - df.groupby("train_category")["has_delay"].mean()) * 100
print("Punctuality Rate by Train Category (%):")
print(on_time_by_type.round(2))
print("\n")

# 5️⃣ Create a pivot table for hour × category on-time rates
pivot_on_time = (1 - pd.pivot_table(
    df,
    values="has_delay",
    index="depart_hour_bucket",
    columns="train_category",
    aggfunc="mean"
)) * 100
print("Punctuality Rate by Hour × Train Category (%):")
print(pivot_on_time.round(2))
print("\n")

# 6️⃣ Plot hourly on-time rate (line chart)
plt.figure(figsize=(10, 5))
plt.plot(on_time_by_hour.index, on_time_by_hour.values, marker='o', color='seagreen')
plt.title("Train Punctuality by Hour", fontsize=14)
plt.xlabel("Hour of the Day (1–24)")
plt.ylabel("Punctuality Rate (%)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(range(0, 25, 2))
plt.tight_layout()
plt.show()

# 7️⃣ Plot on-time rate by train category (bar chart)
plt.figure(figsize=(7, 5))
plt.bar(on_time_by_type.index, on_time_by_type.values, color=['teal', 'orange', 'tomato'])
plt.title("Average Punctuality by Train Category", fontsize=14)
plt.xlabel("Train Category")
plt.ylabel("Punctuality (%)")
plt.grid(axis='y', linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
