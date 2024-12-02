from scipy.io import arff
import numpy as np
import os

# Parameters
input_file = 'data/Scenario A1-ARFF/TimeBasedFeatures-Dataset-15s-VPN.arff'
n_buckets = 3  # Number of buckets
alpha = np.ones(n_buckets)  # Dirichlet parameters
sampling_multiplier = 0.5  # Adjust to sample 50% of the data in each bucket

# Read the .arff file
data, meta = arff.loadarff(input_file)

# Convert data to a list for easier manipulation
data_rows = [list(row) for row in data]
attributes = meta.names()  # Attribute names
attribute_types = meta.types()  # Attribute types
total_rows = len(data_rows)

# Generate bucket probabilities and counts
bucket_probs = np.random.dirichlet(alpha)
total_sampled_rows = int(total_rows * sampling_multiplier)  # Use 50% of the total rows
row_counts = np.round(bucket_probs * total_sampled_rows).astype(int)

print("Bucket Probabilities:", bucket_probs)
print("Row Counts per Bucket:", row_counts)

# Split data into buckets with 50% random sampling
buckets = []

for count in row_counts:
    count = min(count, total_rows)  # Ensure count does not exceed total rows
    indices = np.random.choice(total_rows, size=count, replace=False)  # Sample indices
    sampled_rows = [data_rows[i] for i in indices]  # Retrieve rows by sampled indices
    # Randomly sample 50% of the sampled rows
    sample_size = max(1, len(sampled_rows) // 2)  # Ensure at least 1 row is selected
    sampled_indices = np.random.choice(len(sampled_rows), size=sample_size, replace=False)
    sampled_50_percent = [sampled_rows[i] for i in sampled_indices]
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
