from scipy.io import arff
import numpy as np
import os

# Parameters
input_file = 'data/Scenario A1-ARFF/TimeBasedFeatures-Dataset-60s-VPN.arff'
n_buckets = 10  # Number of buckets
alpha = np.ones(n_buckets) * 2  # Dirichlet parameters for balanced probabilities
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

# Generate bucket probabilities using Dirichlet distribution
bucket_probs = np.random.dirichlet(alpha)
# Scale probabilities to ensure they sum to 1
bucket_probs /= bucket_probs.sum()

# Calculate initial row counts per bucket
raw_row_counts = bucket_probs * total_sampled_rows
row_counts = np.floor(raw_row_counts).astype(int)

# Calculate remaining rows after initial allocation
remaining_rows = total_sampled_rows - row_counts.sum()

# Distribute the remaining rows based on the highest fractional parts
if remaining_rows > 0:
    fractional_parts = raw_row_counts - row_counts
    sorted_indices = np.argsort(fractional_parts)[::-1]  # Descending order
    for i in range(remaining_rows):
        row_counts[sorted_indices[i % n_buckets]] += 1

# Ensure each bucket has at least one row (if possible)
if total_sampled_rows >= n_buckets:
    for i in range(n_buckets):
        if row_counts[i] == 0:
            # Find the bucket with the maximum rows to take one from
            donor = np.argmax(row_counts)
            if row_counts[donor] > 1:
                row_counts[donor] -= 1
                row_counts[i] += 1
            else:
                # If no bucket can donate, leave it as zero
                print(f"Warning: Unable to assign at least one row to bucket {i+1}.")
else:
    print("Warning: Total sampled rows less than number of buckets. Some buckets will be empty.")
    # Assign one row to as many buckets as possible
    row_counts = np.zeros(n_buckets, dtype=int)
    row_counts[:total_sampled_rows] = 1

print("Bucket Probabilities:", bucket_probs)
print("Row Counts per Bucket:", row_counts)

# Shuffle all data indices to ensure random distribution
all_indices = np.arange(total_rows)
np.random.shuffle(all_indices)

# If sampling_multiplier < 1.0, select only a subset of indices
if not replace_sampling and total_sampled_rows < total_rows:
    sampled_indices = all_indices[:total_sampled_rows]
else:
    sampled_indices = all_indices

# Assign sampled indices to buckets based on row_counts
buckets = [[] for _ in range(n_buckets)]
current_idx = 0
for bucket_idx, count in enumerate(row_counts):
    if count > 0:
        assigned_indices = sampled_indices[current_idx:current_idx + count]
        current_idx += count
        sampled_rows = [data_rows[i] for i in assigned_indices]
        buckets[bucket_idx].extend(sampled_rows)
    else:
        # No data assigned to this bucket
        pass

# Ensure all sampled data has been assigned
if current_idx != total_sampled_rows:
    raise ValueError("Mismatch in total sampled rows and assigned rows.")

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
