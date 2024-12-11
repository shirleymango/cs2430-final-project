import os
import numpy as np
from scipy.io import arff
import random

# Parameters
input_file = 'data/Scenario A1-ARFF/TimeBasedFeatures-Dataset-60s-VPN.arff'
n_buckets = 10  # Number of buckets
selection_percentage = 0.8  # Percentage of data to select from each class portion (0 < p <= 1)
random_seed = 42  # Seed for reproducibility
output_dir = 'buckets_class_balanced_selected_order_preserved'  # Output directory for selected buckets

# Set the random seed for reproducibility
np.random.seed(random_seed)
random.seed(random_seed)

# Read the .arff file
print(f"Loading data from {input_file}...")
data, meta = arff.loadarff(input_file)

# Convert data to a list for easier manipulation
data_rows = [list(row) for row in data]
attributes = meta.names()  # Attribute names
attribute_types = meta.types()  # Attribute types
total_rows = len(data_rows)
print(f"Total rows in dataset: {total_rows}")

# Identify the class attribute (assuming the last attribute is the class label)
class_attr = attributes[-1]
class_types = meta[class_attr]
# Extract unique classes
unique_classes = list(set([val.decode() if isinstance(val, bytes) else val for val in data[class_attr]]))
print(f"Unique classes found: {unique_classes}")

# Organize data by class while preserving order
class_data = {cls: [] for cls in unique_classes}
for row in data_rows:
    label = row[-1]
    label = label.decode() if isinstance(label, bytes) else label
    class_data[label].append(row)

# Verify that each class has enough samples
for cls, samples in class_data.items():
    if len(samples) < n_buckets:
        print(f"Warning: Class '{cls}' has only {len(samples)} samples, which is less than the number of buckets ({n_buckets}). Some buckets may not have samples from this class.")

# Split data per class into n_buckets ordered portions
class_portions = {cls: [] for cls in unique_classes}
for cls, samples in class_data.items():
    portion_size = len(samples) // n_buckets
    for i in range(n_buckets):
        start_idx = i * portion_size
        # For the last bucket, include any remaining samples
        if i == n_buckets - 1:
            end_idx = len(samples)
        else:
            end_idx = (i + 1) * portion_size
        portion = samples[start_idx:end_idx]
        class_portions[cls].append(portion)

# Initialize buckets
buckets = [[] for _ in range(n_buckets)]

# Process each class and assign selected samples to buckets while preserving order
for cls in unique_classes:
    print(f"\nProcessing class '{cls}'...")
    for bucket_idx in range(n_buckets):
        portion = class_portions[cls][bucket_idx]
        portion_length = len(portion)
        if portion_length == 0:
            print(f"  Bucket {bucket_idx + 1}: No samples available for class '{cls}'.")
            continue
        selected_count = int(portion_length * selection_percentage)
        # Ensure at least one sample is selected if possible
        if selected_count == 0 and portion_length > 0:
            selected_count = 1
        # Randomly select samples without replacement
        selected_indices = sorted(random.sample(range(portion_length), selected_count))
        selected_samples = [portion[idx] for idx in selected_indices]
        buckets[bucket_idx].extend(selected_samples)
        print(f"  Bucket {bucket_idx + 1}: Selected {len(selected_samples)} out of {portion_length} samples from class '{cls}'.")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Write buckets to .arff files
for idx, bucket_rows in enumerate(buckets):
    bucket_num = idx + 1
    output_file = os.path.join(output_dir, f'bucket_{bucket_num}.arff')

    with open(output_file, 'w') as f:
        # Write the header
        f.write(f"@RELATION bucket_{bucket_num}\n\n")
        for attr_name, attr_type in zip(attributes, attribute_types):
            # Handle nominal attributes
            if isinstance(attr_type, list):
                if attr_name == class_attr:
                    # Extract unique class labels from the bucket
                    nominal_values = sorted(list(set([row[-1].decode() if isinstance(row[-1], bytes) else row[-1] for row in bucket_rows])))
                    nominal_str = ','.join(nominal_values)
                    f.write(f"@ATTRIBUTE {attr_name} {{{nominal_str}}}\n")
                else:
                    nominal_str = ','.join(attr_type)
                    f.write(f"@ATTRIBUTE {attr_name} {{{nominal_str}}}\n")
            else:
                # Numeric attributes
                if attr_type == "nominal":
                    f.write(f"@ATTRIBUTE class1 {{Non-VPN,VPN}}\n")
                else:
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

print(f"\nClass-balanced selected buckets with order preserved have been saved in the '{output_dir}' directory.")
