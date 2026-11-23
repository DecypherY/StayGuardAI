from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed_bookings.csv"

data = pd.read_csv(DATA_PATH)

hotel_cancellation = data.groupby("hotel")["last_minute_cancel"].mean() * 100

print("\nLast-minute cancellation rate by hotel:")
print(hotel_cancellation.round(2))


plt.figure(figsize=(8, 5))

sns.barplot(x=hotel_cancellation.index, y=hotel_cancellation.values)

plt.title("Last-Minute Cancellation Rate by Hotel Type")
plt.xlabel("Hotel Type")
plt.ylabel("Cancellation Rate (%)")

plt.show()


segment_cancellation = (
    data.groupby(["hotel", "market_segment"])["last_minute_cancel"]
    .mean()
    .mul(100)
    .round(2)
)

print("\nCancellation rate by hotel and market segment:")
print(segment_cancellation)


segment_counts = data.groupby(["hotel", "market_segment"])["last_minute_cancel"].agg(
    ["count", "sum", "mean"]
)

segment_counts["mean"] = segment_counts["mean"] * 100

print("\nBooking counts and cancellation rates:")
print(segment_counts.round(2))


lead_time_by_hotel = data.groupby("hotel")["lead_time"].agg(["mean", "median"])

print("\nLead time by hotel:")
print(lead_time_by_hotel.round(2))


data["lead_time_group"] = pd.cut(
    data["lead_time"],
    bins=[-1, 7, 30, 90, 180, float("inf")],
    labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "180+ days"],
)

lead_time_cancellation = (
    data.groupby("lead_time_group", observed=True)["last_minute_cancel"]
    .mean()
    .mul(100)
    .round(2)
)

print("\nLast-minute cancellation rate by lead-time group:")
print(lead_time_cancellation)


plt.figure(figsize=(9, 5))

sns.barplot(x=lead_time_cancellation.index, y=lead_time_cancellation.values)

plt.title("Last-Minute Cancellation Rate by Lead Time")
plt.xlabel("Days Between Booking and Arrival")
plt.ylabel("Cancellation Rate (%)")

plt.show()


hotel_lead_cancellation = (
    data.groupby(["hotel", "lead_time_group"], observed=True)["last_minute_cancel"]
    .mean()
    .mul(100)
    .round(2)
)

print("\nCancellation rate by hotel and lead-time group:")
print(hotel_lead_cancellation)


hotel_lead_cancellation = hotel_lead_cancellation.unstack()

hotel_lead_cancellation.plot(kind="bar", figsize=(10, 6))

plt.title("Last-Minute Cancellation Rate by Hotel and Lead Time")
plt.xlabel("Hotel Type")
plt.ylabel("Cancellation Rate (%)")
plt.xticks(rotation=0)
plt.legend(title="Lead Time Group")
plt.tight_layout()
plt.show()


deposit_cancellation = data.groupby("deposit_type")["last_minute_cancel"].agg(
    ["count", "sum", "mean"]
)

deposit_cancellation["mean"] *= 100

print("\nCancellation rate by deposit type:")
print(deposit_cancellation.round(2))

plt.figure(figsize=(8, 5))

sns.barplot(data=data, x="deposit_type", y="last_minute_cancel", estimator="mean")

plt.title("Last-Minute Cancellation Rate by Deposit Type")
plt.xlabel("Deposit Type")
plt.ylabel("Cancellation Rate")

plt.show()

customer_cancellation = data.groupby("customer_type")["last_minute_cancel"].agg(
    ["count", "sum", "mean"]
)

customer_cancellation["mean"] *= 100

print("\nCancellation rate by customer type:")
print(customer_cancellation.round(2))


plt.figure(figsize=(9, 5))

sns.barplot(data=data, x="customer_type", y="last_minute_cancel", estimator="mean")

plt.title("Last-Minute Cancellation Rate by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Cancellation Rate")

plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


numeric_data = data.select_dtypes(include="number")

correlation = numeric_data.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(correlation, cmap="coolwarm", annot=False)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

target_correlation = (
    numeric_data.corr()["last_minute_cancel"]
    .drop(["last_minute_cancel", "is_canceled"], errors="ignore")
    .sort_values(key=abs, ascending=False)
)

print("\nCorrelation with last-minute cancellation:")
print(target_correlation.round(3))
