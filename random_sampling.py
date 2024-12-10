from scipy.io import arff
import numpy as np
import os

# Parameters
input_file = 'data/Scenario A1-ARFF/TimeBasedFeatures-Dataset-15s-VPN.arff'
n_buckets = 10  # Number of buckets
alpha = np.ones(n_buckets)  # Dirichlet parameters
sampling_multiplier = 0.5  # Adjust to sample 50% of the data in each bucket
min_flows_per_bucket = 200  # Minimum number of flows per bucket after sampling

# Read the .arff file
data, meta = arff.loadarff(input_file)

# Convert data to a list for easier manipulation
data_rows = [list(row) for row in data]
attributes = meta.names()  # Attribute names
attribute_types = meta.types()  # Attribute types
total_rows = len(data_rows)

# Calculate total sampled rows
total_sampled_rows = int(total_rows * sampling_multiplier)  # Use 50% of the total rows
required_initial = n_buckets * int(np.ceil(min_flows_per_bucket / sampling_multiplier))  # 200 / 0.5 = 400 per bucket

# Ensure that the total sampled rows can accommodate the minimum flows per bucket after sampling
if total_sampled_rows < required_initial:
    raise ValueError(
        f"Total sampled rows ({total_sampled_rows}) are less than the required initial "
        f"flows ({required_initial}) for {n_buckets} buckets. "
        f"Consider increasing the sampling_multiplier or decreasing min_flows_per_bucket."
    )

# Calculate remaining rows after allocating initial minimum flows
remaining_sampled_rows = total_sampled_rows - required_initial

# Generate Dirichlet probabilities for the remaining rows
bucket_probs = np.random.dirichlet(alpha)
remaining_counts = np.round(bucket_probs * remaining_sampled_rows).astype(int)

# Adjust counts to ensure the total matches
difference = remaining_sampled_rows - np.sum(remaining_counts)
if difference > 0:
    remaining_counts[:difference] += 1
elif difference < 0:
    remaining_counts[:abs(difference)] -= 1

# Final initial row counts per bucket
initial_row_counts = np.full(n_buckets, int(np.ceil(min_flows_per_bucket / sampling_multiplier))) + remaining_counts

# Ensure that the total allocated rows do not exceed total_rows
if np.sum(initial_row_counts) > total_rows:
    raise ValueError(
        f"Total allocated rows ({np.sum(initial_row_counts)}) exceed total available rows ({total_rows}). "
        f"Consider adjusting the sampling_multiplier or min_flows_per_bucket."
    )

print("Bucket Probabilities (for remaining flows):", bucket_probs)
print("Initial Row Counts per Bucket (before sampling):", initial_row_counts)

# Split data into buckets with the specified row counts
buckets = []
allocated_indices = set()

for idx, count in enumerate(initial_row_counts):
    # Ensure unique sampling without replacement across all buckets
    available_indices = list(set(range(total_rows)) - allocated_indices)
    if count > len(available_indices):
        raise ValueError(
            f"Bucket {idx + 1} requires {count} flows, but only {len(available_indices)} are available."
        )
    indices = np.random.choice(available_indices, size=count, replace=False)
    allocated_indices.update(indices)
    sampled_rows = [data_rows[i] for i in indices]
    
    # Randomly sample 50% of the sampled rows
    sample_size = max(1, len(sampled_rows) // 2)  # Ensure at least 1 row is selected
    sampled_indices = np.random.choice(len(sampled_rows), size=sample_size, replace=False)
    sampled_50_percent = [sampled_rows[i] for i in sampled_indices]
    
    # Ensure that after sampling, each bucket has at least min_flows_per_bucket flows
    if len(sampled_50_percent) < min_flows_per_bucket:
        raise ValueError(
            f"After sampling, bucket {idx + 1} has only {len(sampled_50_percent)} flows, "
            f"which is less than the required minimum of {min_flows_per_bucket}."
        )
    
    buckets.append(sampled_50_percent)

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

    print(f'Bucket {bucket_num} written to {output_file}')
