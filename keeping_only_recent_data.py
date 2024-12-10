from scipy.io import arff
import numpy as np
import os

# Parameters
input_file = 'data/Scenario A1-ARFF/TimeBasedFeatures-Dataset-120s-VPN.arff'
n_buckets = 10  # Number of buckets
alpha = np.ones(n_buckets)  # Dirichlet parameters
sampling_multiplier = 1.0  # Adjust as needed
recent_fraction = 0.5  # Fraction of most recent data to keep in each bucket

# Read the .arff file
data, meta = arff.loadarff(input_file)

# Convert data to a list for easier manipulation
data_rows = [list(row) for row in data]
attributes = meta.names()  # Attribute names
attribute_types = meta.types()  # Attribute types
total_rows = len(data_rows)

# Generate bucket probabilities and counts
bucket_probs = np.random.dirichlet(alpha)
total_sampled_rows = int(total_rows * sampling_multiplier)
row_counts = np.round(bucket_probs * total_sampled_rows).astype(int)

print("Bucket Probabilities:", bucket_probs)
print("Row Counts per Bucket:", row_counts)

# Sort the data rows by the most recent timestamp (assuming the first column is a timestamp)
# Update this column index if the timestamp is in a different position
timestamp_column_index = 0  # Adjust if the timestamp column is not the first column
data_rows.sort(key=lambda x: x[timestamp_column_index])

# Split data into buckets while keeping only the most recent data points
buckets = []

for count in row_counts:
    count = min(count, total_rows)  # Ensure count does not exceed total rows
    # Select the most recent `count` rows
    recent_rows = data_rows[-count:]
    # Keep only the most recent fraction of these rows
    recent_count = max(1, int(len(recent_rows) * recent_fraction))  # Ensure at least one row
    recent_rows = recent_rows[-recent_count:]  # Keep the last `recent_count` rows
    buckets.append(recent_rows)

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