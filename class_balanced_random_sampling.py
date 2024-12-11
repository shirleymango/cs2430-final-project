# reduction_sampling.py

import os
import numpy as np
import pandas as pd
from scipy.io import arff
import argparse

def reduce_sampling(input_file, sampling_multipliers, sampling_fraction, min_flows_per_bucket, output_base_dir, seed=None):
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
    
    for multiplier in sampling_multipliers:
        print(f"\n=== Processing Sampling Multiplier: {multiplier} ===")
        
        # Calculate total sampled rows
        total_sampled_rows = int(total_rows * multiplier)  # e.g., 1.0 for 100%
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
    parser.add_argument('--sampling_fraction', type=float, default=0.5, help='Fraction of data to retain in each bucket after reduction')
    parser.add_argument('--min_flows_per_bucket', type=int, default=10, help='Minimum number of flows per bucket after sampling')
    parser.add_argument('--output_base_dir', type=str, default='reduced_buckets', help='Base directory to save reduced buckets')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    for multiplier in args.sampling_multipliers:
        print(f"\nStarting reduction sampling for multiplier: {multiplier}")
        try:
            # Calculate total_sampled_rows and other parameters per multiplier
            # As the function reduce_sampling currently processes all multipliers in a loop,
            # adjust the function to process one multiplier at a time.
            # To fix the earlier issue, let's refactor the function to handle one multiplier per call.
            # Therefore, modify the reduce_sampling function accordingly.
            # But for simplicity, let's call it once per multiplier in the main loop.
            
            # Modify reduce_sampling to handle one multiplier at a time
            # This requires adjusting the function to take a single multiplier instead of a list
            # Let's redefine the function accordingly
            
            # Here's the adjusted code:
            reduce_sampling_single_multiplier(
                input_file=args.input_file,
                multiplier=multiplier,
                sampling_fraction=args.sampling_fraction,
                min_flows_per_bucket=args.min_flows_per_bucket,
                output_base_dir=args.output_base_dir,
                seed=args.seed
            )
        except Exception as e:
            print(f"Error with sampling_multiplier={multiplier}: {e}")
            continue

def reduce_sampling_single_multiplier(input_file, multiplier, sampling_fraction, min_flows_per_bucket, output_base_dir, seed=None):
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
    
    print(f"\n=== Processing Sampling Multiplier: {multiplier} ===")
    
    # Calculate total sampled rows
    total_sampled_rows = int(total_rows * multiplier)  # e.g., 1.0 for 100%
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduction Sampling for Continual Learning")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input ARFF file')
    parser.add_argument('--sampling_multipliers', type=float, nargs='+', required=True, help='List of sampling multipliers to run')
    parser.add_argument('--sampling_fraction', type=float, default=0.5, help='Fraction of data to retain in each bucket after reduction')
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
                sampling_fraction=args.sampling_fraction,
                min_flows_per_bucket=args.min_flows_per_bucket,
                output_base_dir=args.output_base_dir,
                seed=args.seed
            )
        except Exception as e:
            print(f"Error with sampling_multiplier={multiplier}: {e}")
            continue
