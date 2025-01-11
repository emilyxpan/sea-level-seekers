import pandas as pd
import os

# Directories
csv_directory = r"iharp_training_dataset\Flooding_Data"
file_directory = r"iharp_training_dataset\Copernicus_ENA_Satelite_Maps_Training_Data"

# Date range
date_range = pd.date_range(start="1993-01-01", end="2013-12-31")

# Missing dates from CSVs
csv_missing_dates = set()

# Ensure the CSV directory exists
if not os.path.exists(csv_directory):
    print(f"Directory {csv_directory} does not exist.")
else:
    # Process each CSV file
    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_directory, filename)
            print(f"\nProcessing file: {filename}")

            try:
                # Load the dataset
                data = pd.read_csv(file_path)

                # Ensure the 't' column is in datetime format
                if 't' in data.columns:
                    data['t'] = pd.to_datetime(data['t'], errors='coerce')

                    # Check for duplicate dates
                    duplicate_dates = data[data.duplicated(subset=['t'], keep=False)]

                    # Check for missing dates
                    file_missing_dates = date_range.difference(data['t'])
                    csv_missing_dates.update(file_missing_dates)

                    # Print results for duplicates
                    if not duplicate_dates.empty:
                        print("Duplicate Dates Found:")
                        print(duplicate_dates)
                    else:
                        print("No duplicate dates found.")
                else:
                    print(f"Column 't' not found in {filename}. Skipping.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Missing dates from file existence check
file_missing_dates = set()

# Ensure the file directory exists
if not os.path.exists(file_directory):
    print(f"Directory {file_directory} does not exist.")
else:
    # Check each date in the range for expected files
    for date in date_range:
        # Format the date as YYYYMMDD
        date_str = date.strftime('%Y%m%d')
        # Construct the expected filename
        expected_filename = f"dt_ena_{date_str}_vDT2021.nc"
        # Check if the file exists
        if not os.path.exists(os.path.join(file_directory, expected_filename)):
            file_missing_dates.add(date)

# Compare the missing dates
if csv_missing_dates == file_missing_dates:
    print("\nThe missing dates from both checks match perfectly!")
else:
    print("\nDiscrepancies found between the missing dates.")

    print("\nDates in CSV missing but not in file check:")
    print(sorted(csv_missing_dates - file_missing_dates))

    print("\nDates in file check but not in CSV missing:")
    print(sorted(file_missing_dates - csv_missing_dates))

# Save results to files for further analysis
with open("csv_missing_dates.txt", "w") as f:
    for date in sorted(csv_missing_dates):
        f.write(date.strftime('%Y-%m-%d') + "\n")

with open("file_missing_dates.txt", "w") as f:
    for date in sorted(file_missing_dates):
        f.write(date.strftime('%Y-%m-%d') + "\n")

print("\nMissing dates saved to 'csv_missing_dates.txt' and 'file_missing_dates.txt'.")