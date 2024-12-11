import os
import numpy as np
from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import psutil
import time
import tracemalloc
import csv

# Parameters
bucket_dir = 'buckets_recent_selection'
memory_size = 500  # Maximum rehearsal memory size
rehearsal_memory = []  # Memory for past data
k = 2  # Number of neighbors for kNN
accuracies = []  # Store accuracy after each bucket
metrics = []  # Store computational and memory metrics

# Initialize kNN classifier
knn_model = KNeighborsClassifier(n_neighbors=k)
scaler = StandardScaler()  # Standardize features

# Initialize CPU time tracking
process = psutil.Process(os.getpid())
cpu_times_start = process.cpu_times()

# Load and sort buckets to ensure proper sequential processing
buckets = sorted(
    [os.path.join(bucket_dir, f) for f in os.listdir(bucket_dir) if f.endswith('.arff')],
    key=lambda x: os.path.basename(x)
)

# Preload all bucket data to facilitate accessing the next bucket for testing
bucket_data = []
for bucket_file in buckets:
    print(f"Loading {bucket_file}...")
    
    # Measure loading time and memory
    mem_before = process.memory_info().rss / (1024 * 1024)  # in MB
    start_load = time.time()
    
    # Load data from the .arff file
    data, meta = arff.loadarff(bucket_file)
    data_array = np.array([list(row) for row in data])

    load_time = time.time() - start_load
    mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
    mem_diff = mem_after - mem_before
    
    # Handle empty bucket
    if len(data_array) == 0:
        print(f"Skipping {bucket_file} (empty bucket).")
        bucket_data.append(None)  # Placeholder for empty bucket
        continue

    # Map the nominal class labels to numeric values
    label_map = {b'Non-VPN': 0, b'VPN': 1}  # Map class labels to 0 and 1
    y = np.array([label_map[label] for label in data_array[:, -1]])  # Extract and convert labels
    X = data_array[:, :-1].astype(np.float32)  # Extract features (all columns except the last)

    # Store the raw features and labels for later processing
    bucket_data.append((X, y))
    
    # Log metrics for loading
    metrics.append({
        'Method': 'kNN',
        'Bucket': os.path.basename(bucket_file),
        'Operation': 'Load',
        'Time_Sec': load_time,
        'Memory_MB': mem_diff,
        'CPU_Usage_Percent': None  # Not tracking per-operation CPU usage
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
        'Method': 'kNN',
        'Bucket': os.path.basename(buckets[i]),
        'Operation': 'Scaling',
        'Time_Sec': scale_time,
        'Memory_MB': mem_diff,
        'CPU_Usage_Percent': None
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
        'Method': 'kNN',
        'Bucket': os.path.basename(buckets[i]),
        'Operation': 'Training',
        'Time_Sec': train_time,
        'Memory_MB': mem_train_diff,
        'CPU_Usage_Percent': None
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
        'Method': 'kNN',
        'Bucket': os.path.basename(buckets[i]),
        'Operation': 'Prediction',
        'Time_Sec': pred_time,
        'Memory_MB': mem_diff_pred,
        'CPU_Usage_Percent': None
    })
    
    # Log Accuracy as a separate entry
    metrics.append({
        'Method': 'kNN',
        'Bucket': os.path.basename(buckets[i]),
        'Operation': 'Accuracy',
        'Time_Sec': 0,  # Time taken to compute accuracy is negligible
        'Memory_MB': 0,  # Memory change is negligible
        'CPU_Usage_Percent': None,
        'Accuracy': accuracy  # Additional field for accuracy
    })
    
    print(f"Accuracy on {test_bucket_file} after training on {buckets[i]}: {accuracy:.4f}\n")

# Stop tracking overall memory usage
overall_end_time = time.time()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# Calculate CPU time
cpu_times_end = process.cpu_times()
cpu_time_user = cpu_times_end.user - cpu_times_start.user
cpu_time_system = cpu_times_end.system - cpu_times_start.system
total_cpu_time = cpu_time_user + cpu_time_system

# Calculate and print the average accuracy
if accuracies:
    average_accuracy = np.mean(accuracies)
    final_accuracy = accuracies[-1]
    print(f"\nAverage Accuracy Across All Test Buckets: {average_accuracy:.4f}")
    print(f"Final Accuracy after last training step: {final_accuracy:.4f}")
else:
    print("\nNo accuracies were computed.")

# Save the rehearsal memory for future use
if rehearsal_memory:
    np.savez('knn_rehearsal_memory.npz', X_train=X_train, y_train=y_train)
    print("Rehearsal memory saved!")
else:
    print("Rehearsal memory is empty; nothing was saved.")

# Calculate total time and peak memory
total_time = overall_end_time - overall_start_time
peak_memory = peak / (1024 * 1024)  # Convert to MB

print(f"Total Continual Learning Time: {total_time:.2f} seconds")
print(f"Peak Memory Usage: {peak_memory:.2f} MB")
print(f"Total CPU Time: {total_cpu_time:.2f} seconds")

# Save metrics to a CSV file
metrics_file = 'continual_learning_metrics.csv'
with open(metrics_file, mode='w', newline='') as csv_file:
    fieldnames = ['Method', 'Bucket', 'Operation', 'Time_Sec', 'Memory_MB', 'CPU_Usage_Percent', 'Accuracy']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for metric in metrics:
        # Ensure that 'Accuracy' field exists; set to None if not
        if 'Accuracy' not in metric:
            metric['Accuracy'] = None
        writer.writerow(metric)

print(f"Metrics saved to {metrics_file}")
