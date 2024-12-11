import arff
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuration
input_arff_file = '/Users/tajgulati/cs243_shared/Project_Clean_Code/data/Scenario A1-ARFF/TimeBasedFeatures-Dataset-60s-VPN.arff'   # Path to your ARFF file
output_arff_file = 'output_pca.arff'
variance_to_keep = 0.95  # Keep components that explain 95% of variance

# 1. Read ARFF
with open(input_arff_file, 'r') as f:
    data = arff.load(f)

# Extract attributes and data
attributes = data['attributes']
class_attr_name = attributes[-1][0]
class_attr_values = attributes[-1][1] if isinstance(attributes[-1][1], list) else None

# Convert ARFF data to a Pandas DataFrame
df = pd.DataFrame(data['data'], columns=[attr[0] for attr in attributes])

# 2. Separate numeric features and class
if class_attr_values is not None:
    # Last attribute is class
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
else:
    # No nominal class attribute found, handle accordingly
    # For this script, we assume the last attribute is the class
    raise ValueError("No nominal class attribute found as expected.")

# Convert to numeric (in case they are read as strings) and handle missing or placeholder (-1)
# Identify numeric columns (should be all except last)
numeric_cols = X.select_dtypes(include=[np.number]).columns
X_numeric = X[numeric_cols]

# 3. Handle missing values or placeholders
# Replace -1 with NaN
X_numeric = X_numeric.replace(-1, np.nan)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_numeric)

# 4. Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 5. Apply PCA
pca = PCA(n_components=variance_to_keep)
X_pca = pca.fit_transform(X_scaled)

# X_pca now contains the principal components
# The number of components chosen by PCA can be checked by:
# pca.n_components_

# 6. Prepare output DataFrame
pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pca_columns)

# Add the class column back
df_pca[class_attr_name] = y.values

# 7. Prepare ARFF output
# Construct attributes for ARFF
pca_attributes = [(col, 'NUMERIC') for col in pca_columns]
class_attribute = (class_attr_name, class_attr_values)

new_attributes = pca_attributes + [class_attribute]

# Convert df_pca back to list of lists for ARFF
arff_data = df_pca.values.tolist()

arff_dict = {
    'description': 'PCA transformed data',
    'relation': 'pca_transformed',
    'attributes': new_attributes,
    'data': arff_data
}

# 8. Write to ARFF
with open(output_arff_file, 'w') as f:
    arff.dump(arff_dict, f)

print(f"PCA transformation complete. Output written to {output_arff_file}")

import matplotlib.pyplot as plt

# pca.explained_variance_ratio_ gives the proportion of variance explained by each PC
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Plot a scree plot
plt.figure(figsize=(8,6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_ * 100, alpha=0.7)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# Optionally, plot cumulative variance
cumulative_variance = pca.explained_variance_ratio_.cumsum()
plt.figure(figsize=(8,6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, marker='o')
plt.ylabel('Cumulative Explained Variance (%)')
plt.xlabel('Number of Principal Components')
plt.title('Cumulative Explained Variance')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Assume the first two principal components are PC1 and PC2
PC1 = df_pca['PC1']
PC2 = df_pca['PC2']
labels = df_pca['class1']  # Replace 'class1' with your class column name

# Create a scatter plot colored by class
plt.figure(figsize=(8,6))
for label in labels.unique():
    plt.scatter(PC1[labels == label], PC2[labels == label], label=label, alpha=0.7)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA: First Two Principal Components')
plt.legend()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

PC1 = df_pca['PC1']
PC2 = df_pca['PC2']
PC3 = df_pca['PC3']

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

for label in labels.unique():
    ax.scatter(PC1[labels == label], PC2[labels == label], PC3[labels == label], label=label, alpha=0.7)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D PCA Visualization')
ax.legend()
plt.show()
