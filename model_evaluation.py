import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.io import arff

# Define evaluation function
def evaluate_knn(model, X_test, y_test, scaler):
    # Normalize the test features
    X_test = scaler.transform(X_test)

    # Perform predictions
    predictions = model.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1

# Load rehearsal memory
print("Loading rehearsal memory...")
rehearsal_memory = np.load('knn_rehearsal_memory.npz')
X_train = rehearsal_memory['X_train']
y_train = rehearsal_memory['y_train']

# Initialize kNN model and scaler
k = 5  # Number of neighbors
knn_model = KNeighborsClassifier(n_neighbors=k)
scaler = StandardScaler()

# Fit the kNN model with the rehearsal memory
print("Fitting kNN model with rehearsal memory...")
scaler.fit(X_train)  # Fit scaler on the training data
X_train = scaler.transform(X_train)  # Normalize training data
knn_model.fit(X_train, y_train)

# Load test dataset
test_file = 'bucket_10.arff'  # Replace with your test dataset path
print(f"Loading test dataset from {test_file}...")
test_data, test_meta = arff.loadarff(test_file)

# Convert to NumPy arrays
test_data_array = np.array([list(row) for row in test_data])
label_map = {b'Non-VPN': 0, b'VPN': 1}  # Map class labels to numeric values
X_test = test_data_array[:, :-1].astype(np.float32)  # Features
y_test = np.array([label_map[label] for label in test_data_array[:, -1]])  # Labels

# Evaluate the kNN model
print("Evaluating kNN model on test data...")
evaluate_knn(knn_model, X_test, y_test, scaler)

# Example: Predict on new data
def predict_knn(model, X_new, scaler):
    # Normalize new data
    X_new = scaler.transform(X_new)
    return model.predict(X_new)

# Example usage
print("Example prediction on new data...")
new_data = np.random.rand(1, X_train.shape[1])  # Example new data point
prediction = predict_knn(knn_model, new_data, scaler)
print(f"Prediction for new data: {prediction}")