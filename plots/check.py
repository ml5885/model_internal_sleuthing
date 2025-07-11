import os
import pandas as pd

def check_empty_csvs(base_dirs):
    """
    Checks for empty CSV result files in the given base directories.
    An empty CSV is one that is either 0 bytes or has a header but no data rows.
    """
    print("Checking for empty CSV files...")
    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            print(f"Warning: Directory not found, skipping: {base_dir}")
            continue
        
        for folder_name in sorted(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, folder_name)
            if os.path.isdir(folder_path):
                for csv_filename in ["lexeme_results.csv", "inflection_results.csv"]:
                    csv_path = os.path.join(folder_path, csv_filename)
                    if os.path.exists(csv_path):
                        # Check if file size is very small (e.g., empty or just header)
                        if os.path.getsize(csv_path) < 10: # Arbitrary small size
                            print(f"Found potentially empty file (size < 10 bytes): {folder_path} -> {csv_filename}")
                            continue
                        
                        # Check for no data rows using pandas
                        try:
                            df = pd.read_csv(csv_path)
                            if df.empty:
                                print(f"Found empty CSV (no data rows): {folder_path} -> {csv_filename}")
                        except pd.errors.EmptyDataError:
                            print(f"Found empty CSV (pandas EmptyDataError): {folder_path} -> {csv_filename}")
                        except Exception as e:
                            print(f"Error reading {csv_path}: {e}")

if __name__ == "__main__":
    # The script is in plots/, so we go up one level to find output/
    probe_dirs = [
        os.path.join("..", "output", "probes"),
        os.path.join("..", "output", "probes2")
    ]
    check_empty_csvs(probe_dirs)
    print("Check complete.")
