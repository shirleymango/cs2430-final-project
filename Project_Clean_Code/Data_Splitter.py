from scipy.io import arff
import numpy as np
import os

# Parameters
input_file = '/Users/tajgulati/cs243_shared/Project_Clean_Code/output_pca.arff'
n_buckets = 10  # Number of buckets
sampling_multiplier = 1.0  # Adjust as needed (1.0 means use all data)

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

print("Row Counts per Bucket:", row_counts)

# Select the subset of data based on sampling_multiplier
if sampling_multiplier < 1.0:
    sampled_indices = np.arange(total_sampled_rows)
    sampled_rows = [data_rows[i] for i in sampled_indices]
else:
    sampled_rows = data_rows[:total_sampled_rows]

# Assign sampled rows to buckets sequentially
buckets = [[] for _ in range(n_buckets)]
current_idx = 0
for bucket_idx, count in enumerate(row_counts):
    if count > 0:
        assigned_rows = sampled_rows[current_idx:current_idx + count]
        current_idx += count
        buckets[bucket_idx].extend(assigned_rows)
    else:
        # No data assigned to this bucket
        pass

# Ensure all sampled data has been assigned
if current_idx != total_sampled_rows:
    raise ValueError("Mismatch in total sampled rows and assigned rows.")

print("Row Counts per Bucket after Assignment:", [len(bucket) for bucket in buckets])

# Ensure output directory exists
output_dir = 'buckets'
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
for idx, bucket_rows in enumerate(buckets):
    print(f'Bucket {idx + 1}: {len(bucket_rows)} rows')
