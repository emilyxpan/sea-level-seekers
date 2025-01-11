import pandas as pd
import os

print(os.getcwd())

# File path
file_path = "iharp_training_dataset/cleaned_flooding/atlantic_city"

# Load the dataset
data = pd.read_csv(file_path)

# Ensure the 't' column is in datetime format
data['t'] = pd.to_datetime(data['t'])

# Check for duplicate dates
duplicate_dates = data[data.duplicated(subset=['t'], keep=False)]

# Generate the full range of dates from 1993-01-01 to 2013-12-31
full_date_range = pd.date_range(start="1993-01-01", end="2013-12-31")

print(len(full_date_range))

# Check for missing dates
missing_dates = full_date_range.difference(data['t'])

# Print results
if not duplicate_dates.empty:
    print("Duplicate Dates Found:")
    print(duplicate_dates)
else:
    print("No duplicate dates found.")

if len(missing_dates) > 0:
    print("\nMissing Dates:")
    print(missing_dates)
else:
    print("\nNo missing dates found.")

# import os
# import pandas as pd

# # Directory containing the files
# directory = r"iharp_training_dataset\Copernicus_ENA_Satelite_Maps_Training_Data"

# # Generate the full date range
# date_range = pd.date_range(start="1993-01-01", end="2013-12-31")

# # Keep track of missing dates
# missing_dates = []

# # Check each date in the range
# for date in date_range:
#     # Format the date as YYYYMMDD to match the filenames
#     date_str = date.strftime('%Y%m%d')
#     # Construct the expected filename
#     expected_filename = f"dt_ena_{date_str}_vDT2021.nc"
#     # Check if the file exists
#     if not os.path.exists(os.path.join(directory, expected_filename)):
#         missing_dates.append(date)

# # Print results
# if missing_dates:
#     print("Missing files for the following dates:")
#     for missing_date in missing_dates:
#         print(missing_date.strftime('%Y-%m-%d'))
# else:
#     print("All files are present!")