from scipy.io import arff
import numpy as np
import os
import pandas as pd

# Parameters
input_file = 'data/Scenario A1-ARFF/TimeBasedFeatures-Dataset-15s-VPN.arff'
n_buckets = 5  # Number of buckets
alpha = np.ones(n_buckets)  # Dirichlet parameters
sampling_multiplier = 1.0  # Adjust as needed
sampling_fraction = 0.5  # Fraction of data to retain in each bucket (50%)

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

# Split data into buckets
buckets = []

for count in row_counts:
    count = min(count, total_rows)  # Ensure count does not exceed total rows
    indices = np.random.choice(total_rows, size=count, replace=False)  # Sample indices
    sampled_rows = [data_rows[i] for i in indices]  # Retrieve rows by sampled indices
    buckets.append(sampled_rows)

# Reduce each bucket to the specified fraction while ensuring class balance
reduced_buckets = []
for bucket_rows in buckets:
    # Convert bucket to DataFrame for easier manipulation
    bucket_df = pd.DataFrame(bucket_rows, columns=attributes)
    bucket_df['class1'] = bucket_df['class1'].str.decode('utf-8')  # Decode bytes to strings
    
    # Perform class-balanced random sampling
    classes = bucket_df['class1'].unique()
    sampled_rows = []
    for cls in classes:
        cls_rows = bucket_df[bucket_df['class1'] == cls]
        sample_size = int(len(cls_rows) * sampling_fraction)
        sampled_rows.append(cls_rows.sample(n=sample_size, random_state=42))
    reduced_bucket_df = pd.concat(sampled_rows)
    
    # Append reduced bucket to the list
    reduced_buckets.append(reduced_bucket_df)

# Ensure output directory exists
output_dir = 'buckets_balanced_sampled'
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
