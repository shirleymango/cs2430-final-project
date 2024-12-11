import os
from scipy.io import arff
import numpy as np
import csv

# Parameters
input_file = 'data/Scenario A1-ARFF/TimeBasedFeatures-Dataset-60s-VPN.arff'
n_buckets = 10  # Number of buckets
selection_percentage = 0.8  # Percentage of recent data to keep in each bucket (0 < p <= 1)
output_dir = 'buckets_recent_selection'  # Output directory for selected buckets

# Validate selection_percentage
if not 0 < selection_percentage <= 1:
    raise ValueError("selection_percentage must be between 0 (exclusive) and 1 (inclusive).")

# Read the .arff file
print(f"Loading data from {input_file}...")
data, meta = arff.loadarff(input_file)

# Convert data to a list for easier manipulation, preserving order
data_rows = [list(row) for row in data]
attributes = meta.names()  # Attribute names
attribute_types = meta.types()  # Attribute types
total_rows = len(data_rows)
print(f"Total rows in dataset: {total_rows}")

# Calculate row counts per bucket for even distribution
base_count = total_rows // n_buckets
remainder = total_rows % n_buckets
row_counts = [base_count] * n_buckets

# Distribute the remaining rows to the first 'remainder' buckets
for i in range(remainder):
    row_counts[i] += 1

print("Initial Row Counts per Bucket:", row_counts)

# Split data into ordered buckets
buckets = []
current_idx = 0
for i in range(n_buckets):
    count = row_counts[i]
    bucket = data_rows[current_idx:current_idx + count]
    buckets.append(bucket)
    current_idx += count

# Select only the recent percentage of data in each bucket
selected_buckets = []
for i, bucket in enumerate(buckets):
    bucket_size = len(bucket)
    selected_count = int(np.floor(bucket_size * selection_percentage))
    
    # Ensure at least one row is selected if possible
    if selected_count == 0 and bucket_size > 0:
        selected_count = 1
    
    # Select the last 'selected_count' samples
    selected_samples = bucket[-selected_count:]
    selected_buckets.append(selected_samples)
    
    print(f'Bucket {i + 1}: Retained {len(selected_samples)} out of {bucket_size} rows (Last {selection_percentage*100}%).')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Write selected buckets to .arff files
for idx, bucket_rows in enumerate(selected_buckets):
    bucket_num = idx + 1
    output_file = os.path.join(output_dir, f'bucket_{bucket_num}.arff')

    with open(output_file, 'w') as f:
        # Write the header
        f.write(f"@RELATION bucket_{bucket_num}\n\n")
        for attr_name, attr_type in zip(attributes, attribute_types):
            # Handle nominal attributes
            if isinstance(attr_type, list):
                # Extract unique nominal values present in the bucket for the current attribute
                if attr_type:  # Ensure there are nominal values
                    if any(isinstance(val, bytes) for val in [row[attributes.index(attr_name)] for row in bucket_rows]):
                        # Decode bytes to strings
                        nominal_values = sorted(list(set([
                            row[attributes.index(attr_name)].decode() if isinstance(row[attributes.index(attr_name)], bytes) else row[attributes.index(attr_name)]
                            for row in bucket_rows
                        ])))
                    else:
                        nominal_values = sorted(list(set([
                            row[attributes.index(attr_name)]
                            for row in bucket_rows
                        ])))
                    nominal_str = ','.join(nominal_values)
                    f.write(f"@ATTRIBUTE {attr_name} {{{nominal_str}}}\n")
                else:
                    # If no nominal values are present, default to numeric
                    f.write(f"@ATTRIBUTE {attr_name} NUMERIC\n")
            else:
                # Numeric attributes
                if attr_type == "nominal":
                    f.write(f"@ATTRIBUTE class1 {{Non-VPN,VPN}}\n")
                else:
                    f.write(f"@ATTRIBUTE {attr_name} {attr_type}\n")
        
        # Write the data
        f.write("\n@DATA\n")
        for row in bucket_rows:
            formatted_row = []
            for idx_attr, val in enumerate(row):
                if isinstance(val, bytes):
                    formatted_val = f"'{val.decode()}'"
                elif isinstance(val, str):
                    formatted_val = f"'{val}'"
                else:
                    formatted_val = str(val)
                formatted_row.append(formatted_val)
            f.write(",".join(formatted_row) + "\n")

    print(f'Bucket {bucket_num} written to {output_file} with {len(bucket_rows)} rows.')

# Summary of bucket sizes
print("\nSummary of Selected Bucket Sizes:")
for idx, bucket_rows in enumerate(selected_buckets):
    print(f'Bucket {idx + 1}: {len(bucket_rows)} rows')

print(f"\nRecent selected buckets have been saved in the '{output_dir}' directory.")
