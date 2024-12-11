from scipy.io import arff
import numpy as np
import os
import random

# Parameters
input_file = 'data/Scenario A1-ARFF/TimeBasedFeatures-Dataset-15s-VPN.arff'
n_buckets = 10  # Number of buckets
selection_percentage = 0.8  # Percentage of data to select from each portion (0 < p <= 1)
random_seed = 42  # Seed for reproducibility
sampling_multiplier = 1.0  # Adjust as needed (1.0 means use all data)

# Set the random seed for reproducibility
np.random.seed(random_seed)
random.seed(random_seed)

# Read the .arff file
data, meta = arff.loadarff(input_file)

# Convert data to a list for easier manipulation
data_rows = [list(row) for row in data]
attributes = meta.names()  # Attribute names
attribute_types = meta.types()  # Attribute types
total_rows = len(data_rows)

# Determine total sampled rows
if sampling_multiplier <= 1.0:
    total_sampled_rows = int(total_rows * sampling_multiplier)
    replace_sampling = False
else:
    # Since we're sampling without replacement, set sampling_multiplier to at most 1.0
    total_sampled_rows = total_rows
    replace_sampling = False
    print("Warning: Sampling multiplier > 1.0 is not supported without replacement. Using all available rows.")

# Calculate row counts per bucket for even distribution
base_count = total_sampled_rows // n_buckets
remainder = total_sampled_rows % n_buckets
row_counts = np.array([base_count] * n_buckets)

# Distribute the remaining rows to the first 'remainder' buckets
if remainder > 0:
    row_counts[:remainder] += 1

# Ensure each bucket has at least one row (if possible)
if total_sampled_rows >= n_buckets:
    for i in range(n_buckets):
        if row_counts[i] == 0:
            # Assign one row to this bucket from the end
            row_counts[i] = 1
            row_counts[-1] -= 1
else:
    print("Warning: Total sampled rows less than number of buckets. Some buckets will be empty.")
    # Assign one row to as many buckets as possible
    row_counts = np.zeros(n_buckets, dtype=int)
    row_counts[:total_sampled_rows] = 1

print("Initial Row Counts per Bucket:", row_counts)

# Evenly split the data into n_buckets portions
portion_size = total_sampled_rows // n_buckets
portions = []
for i in range(n_buckets):
    start_idx = i * portion_size
    # For the last bucket, include any remaining rows due to integer division
    if i == n_buckets - 1:
        end_idx = total_sampled_rows
    else:
        end_idx = (i + 1) * portion_size
    portions.append(data_rows[start_idx:end_idx])

# In case total_sampled_rows is not perfectly divisible
# Assign any remaining rows to the last bucket
if total_sampled_rows % n_buckets != 0:
    portions[-1].extend(data_rows[n_buckets * portion_size:total_sampled_rows])

# Initialize buckets
buckets = [[] for _ in range(n_buckets)]

# Process each portion and randomly select a percentage of data to add to the corresponding bucket
for i in range(n_buckets):
    portion = portions[i]
    portion_length = len(portion)
    selected_count = int(portion_length * selection_percentage)
    
    # Ensure at least one row is selected if possible
    if selected_count == 0 and portion_length > 0:
        selected_count = 1
    
    # Randomly select indices without replacement
    selected_indices = random.sample(range(portion_length), selected_count)
    
    # Assign selected rows to the bucket
    selected_rows = [portion[idx] for idx in selected_indices]
    buckets[i].extend(selected_rows)
    
    print(f'Bucket {i + 1}: Selected {len(selected_rows)} out of {portion_length} rows from its portion.')

# Ensure output directory exists
output_dir = 'buckets_selected'
os.makedirs(output_dir, exist_ok=True)

# Write buckets to .arff files
for idx, bucket_rows in enumerate(buckets):
    bucket_num = idx + 1
    output_file = os.path.join(output_dir, f'bucket_{bucket_num}.arff')

    with open(output_file, 'w') as f:
        # Write the header
        f.write(f"@RELATION bucket_{bucket_num}\n\n")
        for attr_name, attr_type in zip(attributes, attribute_types):
            # Handle `class1` explicitly
            if attr_name == 'class1':
                f.write(f"@ATTRIBUTE {attr_name} {{Non-VPN,VPN}}\n")
            elif isinstance(attr_type, list):  # Other categorical attributes
                f.write(f"@ATTRIBUTE {attr_name} {{{','.join(attr_type)}}}\n")
            else:  # Numeric attributes
                f.write(f"@ATTRIBUTE {attr_name} {attr_type}\n")
        
        # Write the data
        f.write("\n@DATA\n")
        for row in bucket_rows:
            formatted_row = [
                f"'{val.decode()}'" if isinstance(val, bytes) else str(val)
                for val in row
            ]
            f.write(",".join(formatted_row) + "\n")

    print(f'Bucket {bucket_num} written to {output_file} with {len(bucket_rows)} rows.')

# Summary of bucket sizes
print("\nSummary of Bucket Sizes:")
for idx, bucket_rows in enumerate(buckets):
    print(f'Bucket {idx + 1}: {len(bucket_rows)} rows')
