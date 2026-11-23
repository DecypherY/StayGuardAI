# Import Path to manage file and folder locations
from pathlib import Path

# Import Pandas for data manipulation and analysis
import pandas as pd

# --------------------------------------------------
# 1. PROJECT PATH CONFIGURATION
# --------------------------------------------------

# Get the absolute path of the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Construct the path to our original dataset
DATA_PATH = ROOT_DIR / "data" / "hotel_bookings.csv"


# --------------------------------------------------
# 2. LOAD THE DATASET
# --------------------------------------------------

# Read the CSV file and store it in a Pandas DataFrame
data = pd.read_csv(DATA_PATH)

# Display the original dataset dimensions
print("Original dataset shape:", data.shape)


# --------------------------------------------------
# 3. REMOVE DUPLICATE RECORDS
# --------------------------------------------------

# Remove exact duplicate rows and create a separate DataFrame
clean_data = data.drop_duplicates()

# Display the number of records after removing duplicates
print("Shape after removing duplicates:", clean_data.shape)


# --------------------------------------------------
# 4. HANDLE MISSING VALUES
# --------------------------------------------------

# Replace missing children values with zero
# Assumption: missing children counts represent no children
clean_data["children"] = clean_data["children"].fillna(0)

# Verify that children has no remaining missing values
print("Missing children values:", clean_data["children"].isna().sum())


# --------------------------------------------------
# 5. CONVERT ARRIVAL MONTHS
# --------------------------------------------------

# Create a dictionary mapping month names to numerical values
MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

# Convert arrival month names into numerical month values
arrival_month = clean_data["arrival_date_month"].map(MONTHS)


# --------------------------------------------------
# 6. CREATE ARRIVAL DATE
# --------------------------------------------------

# Combine year, month and day into a single datetime column
arrival_date = pd.to_datetime(
    {
        "year": clean_data["arrival_date_year"],
        "month": arrival_month,
        "day": clean_data["arrival_date_day_of_month"],
    }
)


# --------------------------------------------------
# 7. CONVERT RESERVATION STATUS DATE
# --------------------------------------------------

# Convert reservation status dates into datetime format
reservation_date = pd.to_datetime(clean_data["reservation_status_date"])


# --------------------------------------------------
# 8. CALCULATE DAYS BEFORE ARRIVAL
# --------------------------------------------------

# Subtract reservation status date from arrival date
# The result is a Pandas Timedelta Series
days_before_arrival = arrival_date - reservation_date

# Extract the number of days from each Timedelta value
days_before_arrival = days_before_arrival.dt.days


# --------------------------------------------------
# 9. CREATE TARGET VARIABLE
# --------------------------------------------------

# Mark bookings cancelled between 0 and 14 days before arrival as 1
# Mark all other bookings as 0
clean_data["last_minute_cancel"] = (
    (clean_data["is_canceled"] == 1)
    & (days_before_arrival >= 0)
    & (days_before_arrival <= 14)
).astype(int)


# --------------------------------------------------
# 10. CHECK TARGET DISTRIBUTION
# --------------------------------------------------

# Display the number of bookings in each target class
print("\nCancellation target counts:")
print(clean_data["last_minute_cancel"].value_counts())

# Display the percentage of bookings in each target class
print("\nCancellation target percentages:")
print(clean_data["last_minute_cancel"].value_counts(normalize=True).mul(100).round(2))


# --------------------------------------------------
# 11. SAVE PROCESSED DATASET
# --------------------------------------------------

# Construct the output path
OUTPUT_PATH = ROOT_DIR / "data" / "processed_bookings.csv"

# Save the processed DataFrame as a CSV file
clean_data.to_csv(OUTPUT_PATH, index=False)

print("\nPreprocessing completed!")
print("Final dataset shape:", clean_data.shape)
print("Saved to:", OUTPUT_PATH)
