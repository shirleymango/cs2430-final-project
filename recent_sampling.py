# reduction_sampling.py

import os
import numpy as np
import pandas as pd
from scipy.io import arff
import argparse

def reduce_sampling_single_multiplier(input_file, multiplier, recent_fraction, min_flows_per_bucket, output_base_dir, seed=None):
    """
    Perform reduction sampling for a single sampling multiplier.

    Parameters:
        input_file (str): Path to the input ARFF file.
        multiplier (float): Sampling multiplier (e.g., 1.0 for 100%).
        recent_fraction (float): Fraction of most recent data to keep in each bucket.
        min_flows_per_bucket (int): Minimum number of flows per bucket after sampling.
        output_base_dir (str): Base directory to save reduced buckets.
        seed (int, optional): Random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Parameters
    n_buckets = 10  # Number of buckets
    alpha = np.ones(n_buckets)  # Dirichlet parameters
    
    # Read the .arff file
    data, meta = arff.loadarff(input_file)
    
    # Convert data to a list for easier manipulation
    data_rows = [list(row) for row in data]
    attributes = meta.names()  # Attribute names
    attribute_types = meta.types()  # Attribute types
    total_rows = len(data_rows)
    
    # Generate bucket probabilities and counts
    bucket_probs = np.random.dirichlet(alpha)
    total_sampled_rows = int(total_rows * multiplier)  # Use 100% of the total rows
    row_counts = np.round(bucket_probs * total_sampled_rows).astype(int)
    
    print(f"\n=== Processing Sampling Multiplier: {multiplier} ===")
    print("Bucket Probabilities:", bucket_probs)
    print("Row Counts per Bucket:", row_counts)
    
    # Sort the data rows by the most recent timestamp (assuming the first column is a timestamp)
    # Update this column index if the timestamp is in a different position
    timestamp_column_index = 0  # Adjust if the timestamp column is not the first column
    data_rows.sort(key=lambda x: x[timestamp_column_index])
    
    # Split data into buckets with the specified row counts
    buckets = []
    allocated_indices = set()
    
    for idx, count in enumerate(row_counts):
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
            # Calculate the number of samples to retain
            sample_size = max(int(len(cls_rows) * recent_fraction), 1)  # Ensure at least one row
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
    
    # Ensure output directory exists for the current sampling_multiplier
    sampling_output_dir = os.path.join(output_base_dir, f'sampling_{multiplier}')
    os.makedirs(sampling_output_dir, exist_ok=True)
    
    # Write reduced buckets to .arff files
    for idx, reduced_bucket_df in enumerate(reduced_buckets):
        bucket_num = idx + 1
        output_file = os.path.join(sampling_output_dir, f'bucket_{bucket_num}.arff')
    
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

def main():
    parser = argparse.ArgumentParser(description="Reduction Sampling for Continual Learning")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input ARFF file')
    parser.add_argument('--sampling_multipliers', type=float, nargs='+', required=True, help='List of sampling multipliers to run')
    parser.add_argument('--recent_fraction', type=float, default=0.5, help='Fraction of most recent data to retain in each bucket after reduction')
    parser.add_argument('--min_flows_per_bucket', type=int, default=10, help='Minimum number of flows per bucket after sampling')
    parser.add_argument('--output_base_dir', type=str, default='reduced_buckets', help='Base directory to save reduced buckets')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    for multiplier in args.sampling_multipliers:
        print(f"\nStarting reduction sampling for multiplier: {multiplier}")
        try:
            reduce_sampling_single_multiplier(
                input_file=args.input_file,
                multiplier=multiplier,
                recent_fraction=args.recent_fraction,
                min_flows_per_bucket=args.min_flows_per_bucket,
                output_base_dir=args.output_base_dir,
                seed=args.seed
            )
        except Exception as e:
            print(f"Error with sampling_multiplier={multiplier}: {e}")
            continue

if __name__ == "__main__":
    main()
