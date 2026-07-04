import os
import mne
import pandas as pd

# Define input and output directories
edf_folder = r"C:\sem 2 thejan\Paper\edf"
csv_output_folder = r"C:\sem 2 thejan\Paper\csv"
error_log_file = os.path.join(csv_output_folder, "error_log.txt")  # Log for failed files

# Ensure the output folder exists
os.makedirs(csv_output_folder, exist_ok=True)

# Find all EDF files in subdirectories
edf_files = []
for root, _, files in os.walk(edf_folder):
    for file in files:
        if file.endswith(".edf"):
            edf_files.append(os.path.join(root, file))

# Check if any EDF files were found
if not edf_files:
    print("❌ No EDF files found. Check the directory structure.")
    exit()

print(f"✅ Found {len(edf_files)} EDF files.")

# Open error log file
with open(error_log_file, "w") as error_log:

    # Process each EDF file
    for edf_path in edf_files:
        edf_file = os.path.basename(edf_path)  # Extract filename
        print(f"🔄 Processing: {edf_file}")

        try:
            # Read EDF file with encoding fix
            raw = mne.io.read_raw_edf(edf_path, preload=True, encoding="latin1")

            # Extract raw data and timestamps
            data, times = raw.get_data(return_times=True)
            channel_names = raw.ch_names

            # Convert to DataFrame
            df = pd.DataFrame(data.T, columns=channel_names)
            df.insert(0, "Time", times)  # Add time column

            # Save to CSV (keeping original folder structure)
            relative_path = os.path.relpath(edf_path, edf_folder)  # Get relative path
            csv_filename = os.path.splitext(relative_path)[0] + ".csv"
            csv_path = os.path.join(csv_output_folder, csv_filename)

            # Ensure subdirectories exist in the output folder
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            # Save to CSV
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved: {csv_path}")

        except Exception as e:
            print(f"⚠️ Error processing {edf_file}: {e}")
            error_log.write(f"{edf_file}: {e}\n")

print("🎉 Conversion completed! Check 'error_log.txt' for any issues.")
