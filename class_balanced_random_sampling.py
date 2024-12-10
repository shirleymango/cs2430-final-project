from scipy.io import arff
import numpy as np
import os
import pandas as pd

# Parameters
input_file = 'data/Scenario A1-ARFF/TimeBasedFeatures-Dataset-15s-VPN.arff'
n_buckets = 10  # Number of buckets
alpha = np.ones(n_buckets)  # Dirichlet parameters
sampling_multiplier = 1.0  # Adjust as needed
sampling_fraction = 0.5  # Fraction of data to retain in each bucket (50%)
min_flows_per_bucket = 10  # Minimum number of flows per bucket after sampling

# Read the .arff file
data, meta = arff.loadarff(input_file)

# Convert data to a list for easier manipulation
data_rows = [list(row) for row in data]
attributes = meta.names()  # Attribute names
attribute_types = meta.types()  # Attribute types
total_rows = len(data_rows)

# Calculate total sampled rows
total_sampled_rows = int(total_rows * sampling_multiplier)  # Use 100% of the total rows
required_minimum = n_buckets * min_flows_per_bucket / sampling_fraction

# Ensure that the total sampled rows can accommodate the minimum flows per bucket after sampling
if total_sampled_rows < required_minimum:
    raise ValueError(
        f"Total sampled rows ({total_sampled_rows}) are less than the required minimum "
        f"flows ({required_minimum}) to achieve {min_flows_per_bucket} flows per bucket after "
        f"sampling fraction ({sampling_fraction}). "
        f"Consider increasing the sampling_multiplier or decreasing min_flows_per_bucket."
    )

# Calculate the number of rows to allocate before sampling
# Each bucket needs at least min_flows_per_bucket / sampling_fraction flows before sampling
initial_min_flows = int(np.ceil(min_flows_per_bucket / sampling_fraction))
required_initial = n_buckets * initial_min_flows

# Check if required_initial exceeds total_sampled_rows
if required_initial > total_sampled_rows:
    raise ValueError(
        f"Required initial flows ({required_initial}) exceed total sampled rows ({total_sampled_rows}). "
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
initial_row_counts = np.full(n_buckets, initial_min_flows) + remaining_counts

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
    buckets.append(sampled_rows)

# Reduce each bucket to the specified fraction while ensuring class balance
reduced_buckets = []
for bucket_idx, bucket_rows in enumerate(buckets):
    # Convert bucket to DataFrame for easier manipulation
    bucket_df = pd.DataFrame(bucket_rows, columns=attributes)
    bucket_df['class1'] = bucket_df['class1'].str.decode('utf-8')  # Decode bytes to strings

    # Perform class-balanced random sampling
    sampled_rows = []
    classes = bucket_df['class1'].unique()
    for cls in classes:
        cls_rows = bucket_df[bucket_df['class1'] == cls]
        # Calculate the number of samples needed to meet the minimum after sampling
        min_required_before_sampling = int(np.ceil(min_flows_per_bucket / sampling_fraction))
        if len(cls_rows) < min_required_before_sampling:
            raise ValueError(
                f"Class '{cls}' in bucket {bucket_idx + 1} does not have enough rows "
                f"({len(cls_rows)}) to satisfy the minimum required after sampling "
                f"({min_flows_per_bucket}). Consider increasing the sampling_multiplier or decreasing min_flows_per_bucket."
            )
        sample_size = max(int(len(cls_rows) * sampling_fraction), 1)
        # Ensure that at least min_flows_per_bucket are retained after sampling
        if sample_size < min_flows_per_bucket:
            sample_size = min_flows_per_bucket
        sampled_cls = cls_rows.sample(n=sample_size, random_state=42)
        sampled_rows.append(sampled_cls)
    reduced_bucket_df = pd.concat(sampled_rows)

    # Verify the minimum flows per bucket
    if len(reduced_bucket_df) < min_flows_per_bucket:
        raise ValueError(
            f"After sampling, bucket {bucket_idx + 1} has only {len(reduced_bucket_df)} flows, "
            f"which is less than the required minimum of {min_flows_per_bucket}."
        )

    reduced_buckets.append(reduced_bucket_df)

# Ensure output directory exists
output_dir = 'buckets'
os.makedirs(output_dir, exist_ok=True)

# Write reduced buckets to .arff files
for idx, reduced_bucket_df in enumerate(reduced_buckets):
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
        for _, row in reduced_bucket_df.iterrows():
            formatted_row = [
                f"'{val}'" if isinstance(val, str) else str(val) for val in row.values
            ]
            f.write(",".join(formatted_row) + "\n")

    print(f'Bucket {bucket_num} written to {output_file}')
