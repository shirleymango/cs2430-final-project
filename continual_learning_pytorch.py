from scipy.io import arff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Parameters
bucket_dir = 'buckets'
memory_size = 500  # Maximum rehearsal memory size
rehearsal_memory = []  # Memory for past data
k = 2  # Number of neighbors for kNN
accuracies = []  # Store accuracy after each bucket

# Load buckets
buckets = [os.path.join(bucket_dir, f) for f in os.listdir(bucket_dir) if f.endswith('.arff')]

# Initialize kNN classifier
knn_model = KNeighborsClassifier(n_neighbors=k)
scaler = StandardScaler()  # Standardize features

# Train incrementally on each bucket
for bucket_file in buckets:
    print(f"Processing {bucket_file}...")
    
    # Load data from the .arff file
    data, meta = arff.loadarff(bucket_file)
    data_array = np.array([list(row) for row in data])

    # Handle empty bucket
    if len(data_array) == 0:
        print(f"Skipping {bucket_file} (empty bucket).")
        continue

    # Map the nominal class labels to numeric values
    label_map = {b'Non-VPN': 0, b'VPN': 1}  # Map class labels to 0 and 1
    y = np.array([label_map[label] for label in data_array[:, -1]])  # Extract and convert labels
    X = data_array[:, :-1].astype(np.float32)  # Extract features (all columns except the last)

    # Normalize features for better training performance
    X = scaler.fit_transform(X)  # Standardize the current bucket

    # Add the current bucket's data to rehearsal memory
    rehearsal_memory.extend(list(zip(X, y)))
    if len(rehearsal_memory) > memory_size:
        rehearsal_memory = rehearsal_memory[-memory_size:]  # Retain the most recent samples

    # Combine rehearsal memory with current bucket
    rehearsal_X, rehearsal_y = zip(*rehearsal_memory)
    X_train = np.array(rehearsal_X)
    y_train = np.array(rehearsal_y)

    # Train the kNN model on the combined data
    knn_model.fit(X_train, y_train)
    print(f"Finished training on {bucket_file}")

    # Compute accuracy on rehearsal memory
    predictions = knn_model.predict(X_train)
    accuracy = accuracy_score(y_train, predictions)
    accuracies.append(accuracy)
    print(f"Accuracy on rehearsal memory after {bucket_file}: {accuracy:.4f}")

# Calculate and print the average accuracy
average_accuracy = np.mean(accuracies)
print(f"\nAverage Accuracy Across All Buckets: {average_accuracy:.4f}")

# Save the rehearsal memory for future use
np.savez('knn_rehearsal_memory.npz', X_train=X_train, y_train=y_train)
print("Rehearsal memory saved!")