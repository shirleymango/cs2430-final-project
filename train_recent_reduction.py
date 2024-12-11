# train_multiple_reduction.py

import os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import psutil
import time
import tracemalloc
import csv
import argparse

def load_bucket_data(bucket_file):
    """
    Load data from a single .arff bucket file.
    
    Parameters:
        bucket_file (str): Path to the bucket .arff file.
        
    Returns:
        tuple or None: (X, y) if data is present, else None.
    """
    data, meta = arff.loadarff(bucket_file)
    data_array = np.array([list(row) for row in data])

    # Handle empty bucket
    if len(data_array) == 0:
        print(f"Skipping {bucket_file} (empty bucket).")
        return None

    # Map the nominal class labels to numeric values
    label_map = {b'Non-VPN': 0, b'VPN': 1}  # Map class labels to 0 and 1
    y = np.array([label_map[label] for label in data_array[:, -1]])  # Extract and convert labels
    X = data_array[:, :-1].astype(np.float32)  # Extract features (all columns except the last)

    return X, y

def process_sampling_multiplier(sampling_dir, sampling_multiplier, memory_size, k, seed=None):
    """
    Process training and evaluation for a single sampling_multiplier.

    Parameters:
        sampling_dir (str): Directory containing reduced bucket .arff files.
        sampling_multiplier (float): The sampling multiplier value.
        memory_size (int): Maximum rehearsal memory size.
        k (int): Number of neighbors for kNN.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        dict: Summary metrics for the sampling_multiplier.
    """
    print(f"\n=== Processing Sampling Multiplier: {sampling_multiplier} ===")
    # Initialize variables
    rehearsal_memory = []  # Memory for past data
    accuracies = []        # Store accuracy after each bucket
    metrics = []           # Store computational and memory metrics

    # Initialize kNN classifier and scaler
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scaler = StandardScaler()

    # Load and sort buckets to ensure proper sequential processing
    buckets = sorted(
        [os.path.join(sampling_dir, f) for f in os.listdir(sampling_dir) if f.endswith('.arff')],
        key=lambda x: os.path.basename(x)
    )

    # Preload all bucket data
    bucket_data = []
    for bucket_file in buckets:
        print(f"Loading {bucket_file}...")
        
        # Measure loading time and memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # in MB
        start_load = time.time()
        
        X_y = load_bucket_data(bucket_file)
        load_time = time.time() - start_load
        mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
        mem_diff = mem_after - mem_before

        if X_y is None:
            bucket_data.append(None)  # Placeholder for empty bucket
            continue

        X, y = X_y
        bucket_data.append((X, y))
        
        # Log metrics for loading
        metrics.append({
            'Sampling_Multiplier': sampling_multiplier,
            'Bucket': os.path.basename(bucket_file),
            'Operation': 'Load',
            'Time_Sec': load_time,
            'Memory_MB': mem_diff,
            'CPU_Usage_Percent': process.cpu_percent(interval=None),
            'Accuracy': None
        })

    print("All buckets loaded.\n")

    # Start tracking overall memory usage
    tracemalloc.start()
    overall_start_time = time.time()

    # Iterate through each bucket except the last one for training and testing
    for i in range(len(bucket_data) - 1):
        current_bucket = bucket_data[i]
        
        # Skip empty buckets
        if current_bucket is None:
            print(f"Skipping training on bucket index {i} (empty bucket).")
            continue
        
        X_current, y_current = current_bucket

        print(f"Processing training on {buckets[i]}...")
        
        # Measure scaling time and memory
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)  # in MB
        start_scale = time.time()
        
        # Normalize features using the scaler fitted on the rehearsal memory plus current bucket
        # Combine rehearsal memory and current bucket for scaling
        if rehearsal_memory:
            combined_X = np.vstack([X_current] + [x for x, _ in rehearsal_memory])
        else:
            combined_X = X_current
        scaler.fit(combined_X)
        X_current_scaled = scaler.transform(X_current)
        
        scale_time = time.time() - start_scale
        mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
        mem_diff = mem_after - mem_before
        
        metrics.append({
            'Sampling_Multiplier': sampling_multiplier,
            'Bucket': os.path.basename(buckets[i]),
            'Operation': 'Scaling',
            'Time_Sec': scale_time,
            'Memory_MB': mem_diff,
            'CPU_Usage_Percent': process.cpu_percent(interval=None),
            'Accuracy': None
        })
        
        # Add the current bucket's data to rehearsal memory
        rehearsal_memory.extend(list(zip(X_current_scaled, y_current)))
        if len(rehearsal_memory) > memory_size:
            rehearsal_memory = rehearsal_memory[-memory_size:]  # Retain the most recent samples
        
        # Combine rehearsal memory for training
        rehearsal_X, rehearsal_y = zip(*rehearsal_memory)
        X_train = np.array(rehearsal_X)
        y_train = np.array(rehearsal_y)
        
        # Measure training time and memory
        start_train = time.time()
        knn_model.fit(X_train, y_train)
        train_time = time.time() - start_train
        mem_train_after = process.memory_info().rss / (1024 * 1024)  # in MB
        mem_train_diff = mem_train_after - mem_after
        
        metrics.append({
            'Sampling_Multiplier': sampling_multiplier,
            'Bucket': os.path.basename(buckets[i]),
            'Operation': 'Training',
            'Time_Sec': train_time,
            'Memory_MB': mem_train_diff,
            'CPU_Usage_Percent': process.cpu_percent(interval=None),
            'Accuracy': None
        })
        
        print(f"Finished training on {buckets[i]}")
        
        # Prepare the next bucket for testing
        test_bucket = bucket_data[i + 1]
        test_bucket_file = buckets[i + 1]
        
        if test_bucket is None:
            print(f"Skipping testing on bucket index {i + 1} (empty bucket).\n")
            continue

        X_test, y_test = test_bucket
        
        # Measure prediction time and memory
        mem_before_pred = process.memory_info().rss / (1024 * 1024)  # in MB
        start_pred = time.time()
        
        # Normalize test features using the same scaler fitted on training data
        X_test_scaled = scaler.transform(X_test)
        
        # Predict and calculate accuracy on the test bucket
        predictions = knn_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        
        pred_time = time.time() - start_pred
        mem_after_pred = process.memory_info().rss / (1024 * 1024)  # in MB
        mem_diff_pred = mem_after_pred - mem_before_pred
        
        metrics.append({
            'Sampling_Multiplier': sampling_multiplier,
            'Bucket': os.path.basename(buckets[i]),
            'Operation': 'Prediction',
            'Time_Sec': pred_time,
            'Memory_MB': mem_diff_pred,
            'CPU_Usage_Percent': process.cpu_percent(interval=None),
            'Accuracy': None
        })
        
        # Log Accuracy as a separate entry
        metrics.append({
            'Sampling_Multiplier': sampling_multiplier,
            'Bucket': os.path.basename(buckets[i]),
            'Operation': 'Accuracy',
            'Time_Sec': 0,  # Time taken to compute accuracy is negligible
            'Memory_MB': 0,  # Memory change is negligible
            'CPU_Usage_Percent': process.cpu_percent(interval=None),
            'Accuracy': accuracy  # Additional field for accuracy
        })
        
        print(f"Accuracy on {test_bucket_file} after training on {buckets[i]}: {accuracy:.4f}\n")
    
    # Stop tracking overall memory usage
    overall_end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate and print the average accuracy
    if accuracies:
        average_accuracy = np.mean(accuracies)
        print(f"\nAverage Accuracy Across All Buckets for Sampling Multiplier {sampling_multiplier}: {average_accuracy:.4f}")
    else:
        print(f"\nNo accuracies were computed for Sampling Multiplier {sampling_multiplier}.")
        average_accuracy = None
    
    # Save the rehearsal memory for future use
    if rehearsal_memory:
        rehearsal_output_dir = 'rehearsal_memories'
        os.makedirs(rehearsal_output_dir, exist_ok=True)
        np.savez(os.path.join(rehearsal_output_dir, f'knn_rehearsal_memory_{sampling_multiplier}.npz'), 
                 X_train=X_train, y_train=y_train)
        print(f"Rehearsal memory saved for Sampling Multiplier {sampling_multiplier}!")
    else:
        print(f"Rehearsal memory is empty for Sampling Multiplier {sampling_multiplier}; nothing was saved.")
    
    # Calculate total time and peak memory
    total_time = overall_end_time - overall_start_time
    peak_memory = peak / (1024 * 1024)  # Convert to MB
    
    print(f"Total Continual Learning Time for Sampling Multiplier {sampling_multiplier}: {total_time:.2f} seconds")
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")
    
    # Save metrics to a CSV file per sampling_multiplier
    metrics_output_dir = 'metrics'
    os.makedirs(metrics_output_dir, exist_ok=True)
    metrics_file = os.path.join(metrics_output_dir, f'continual_learning_metrics_sampling_{sampling_multiplier}.csv')
    with open(metrics_file, mode='w', newline='') as csv_file:
        fieldnames = ['Sampling_Multiplier', 'Bucket', 'Operation', 'Time_Sec', 'Memory_MB', 'CPU_Usage_Percent', 'Accuracy']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric)
    
    print(f"Metrics saved to {metrics_file}")
    
    return {
        'Sampling_Multiplier': sampling_multiplier,
        'Average_Accuracy': average_accuracy,
        'Total_Time_Sec': total_time,
        'Peak_Memory_MB': peak_memory
    }

def train_multiple_reduction(sampled_base_dir, memory_size, k, output_metrics_dir, seed=None):
    """
    Train and evaluate KNN models for multiple sampling multipliers.

    Parameters:
        sampled_base_dir (str): Base directory containing reduced bucket sets.
        memory_size (int): Maximum rehearsal memory size.
        k (int): Number of neighbors for kNN.
        output_metrics_dir (str): Directory to save metrics CSV files.
        seed (int, optional): Random seed for reproducibility.
    """
    # Get list of sampling directories
    sampling_dirs = [
        os.path.join(sampled_base_dir, d) for d in os.listdir(sampled_base_dir)
        if os.path.isdir(os.path.join(sampled_base_dir, d)) and d.startswith('sampling_')
    ]

    if not sampling_dirs:
        print(f"No sampling directories found in {sampled_base_dir}. Exiting.")
        return

    # Extract sampling multipliers from directory names
    sampling_multipliers = []
    for d in sampling_dirs:
        try:
            multiplier = float(d.split('_')[-1])
            sampling_multipliers.append((d, multiplier))
        except ValueError:
            print(f"Unable to extract sampling multiplier from directory name: {d}. Skipping.")

    if not sampling_multipliers:
        print("No valid sampling multipliers found. Exiting.")
        return

    # Aggregate summary metrics for all sampling multipliers
    summary_metrics = []

    for sampling_dir, multiplier in sorted(sampling_multipliers, key=lambda x: x[1]):
        summary = process_sampling_multiplier(
            sampling_dir=sampling_dir,
            sampling_multiplier=multiplier,
            memory_size=memory_size,
            k=k,
            seed=seed
        )
        if summary:
            summary_metrics.append(summary)

    # Save summary metrics to a combined CSV
    if summary_metrics:
        summary_df = pd.DataFrame(summary_metrics)
        summary_output_dir = output_metrics_dir
        os.makedirs(summary_output_dir, exist_ok=True)
        summary_file_csv = os.path.join(summary_output_dir, 'summary_metrics_all_sampling_multipliers.csv')
        summary_df.to_csv(summary_file_csv, index=False)
        print(f"\nSummary metrics for all sampling multipliers saved to {summary_file_csv}")

        # Optionally, save as LaTeX table
        summary_file_latex = os.path.join(summary_output_dir, 'summary_metrics_all_sampling_multipliers.tex')
        try:
            summary_df.to_latex(summary_file_latex, index=False, float_format="%.4f")
            print(f"Summary metrics saved as LaTeX table: {summary_file_latex}")
        except Exception as e:
            print(f"Error saving LaTeX table: {e}")
    else:
        print("No summary metrics to save.")

def main():
    parser = argparse.ArgumentParser(description="Train KNN on Multiple Reduction Sampling Multipliers")
    parser.add_argument('--sampled_base_dir', type=str, default='reduced_buckets', help='Base directory containing reduced bucket sets')
    parser.add_argument('--memory_size', type=int, default=500, help='Maximum rehearsal memory size')
    parser.add_argument('--k', type=int, default=2, help='Number of neighbors for kNN')
    parser.add_argument('--output_metrics_dir', type=str, default='metrics', help='Directory to save metrics CSV files')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    args = parser.parse_args()

    train_multiple_reduction(
        sampled_base_dir=args.sampled_base_dir,
        memory_size=args.memory_size,
        k=args.k,
        output_metrics_dir=args.output_metrics_dir,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
