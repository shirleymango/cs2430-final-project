import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

# Parameters
summary_stats_dir = 'Summary_Stats'  # Directory containing the CSV files
output_dir = 'Graphs'  # Directory to save the generated graphs
os.makedirs(output_dir, exist_ok=True)

# Initialize lists to store data
all_metrics = []

# Regular expression to extract method name and parameters
# Example filenames:
# - continual_learning_metrics_Baseline.csv
# - continual_learning_metrics_Random_0.2.csv
pattern = r'continual_learning_metrics_(.+)\.csv'

# Load all CSV files
for filename in os.listdir(summary_stats_dir):
    if filename.startswith('continual_learning_metrics_') and filename.endswith('.csv'):
        match = re.match(pattern, filename)
        if match:
            method_full_name = match.group(1)  # e.g., 'Baseline' or 'Random_0.2'
            file_path = os.path.join(summary_stats_dir, filename)
            print(f"Loading file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_columns = ['Accuracy', 'Time_Sec', 'Memory_MB']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"  Warning: Missing columns {missing_columns} in file '{filename}'. These columns will be skipped.")
                for col in missing_columns:
                    df[col] = None  # Assign NaN for missing columns
            
            df['Method'] = method_full_name
            all_metrics.append(df)
        else:
            print(f"Filename '{filename}' does not match the expected pattern. Skipping.")

# Concatenate all metrics into a single DataFrame
if not all_metrics:
    raise ValueError("No valid CSV files found in the specified directory.")

metrics_df = pd.concat(all_metrics, ignore_index=True)

# Display the first few rows of the combined DataFrame
print("\nCombined Metrics DataFrame (First 5 Rows):")
print(metrics_df.head())

# Ensure numeric types for these columns
for col in ['Accuracy', 'Time_Sec', 'Memory_MB']:
    if col in metrics_df.columns:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
    else:
        metrics_df[col] = None  # Assign NaN if column is missing

# Extract Data_Amount from Method names if present
metrics_df['Method_Base'] = metrics_df['Method'].apply(lambda x: re.sub(r'_\d+\.?\d*$', '', x))
metrics_df['Data_Amount'] = metrics_df['Method'].apply(lambda x: re.findall(r'_(\d+\.?\d*)$', x))
metrics_df['Data_Amount'] = metrics_df['Data_Amount'].apply(lambda x: float(x[0]) if x else None)

# Verify extraction
print("\nData Amount Extraction (Unique Combinations):")
print(metrics_df[['Method', 'Method_Base', 'Data_Amount']].drop_duplicates())

# Separate rows containing accuracy information and those that don't
accuracy_df = metrics_df[metrics_df['Operation'] == 'Accuracy'].dropna(subset=['Accuracy'])
time_memory_df = metrics_df[metrics_df['Operation'] != 'Accuracy']

# Aggregate Accuracy
accuracy_agg = accuracy_df.groupby('Method')['Accuracy'].mean().reset_index(name='Average_Accuracy')

# Aggregate Time and Memory from rows that are not the Accuracy rows
time_agg = time_memory_df.groupby('Method')['Time_Sec'].mean().reset_index(name='Average_Time_Sec')
memory_agg = time_memory_df.groupby('Method')['Memory_MB'].mean().reset_index(name='Average_Memory_MB')

# Merge all aggregates
aggregate_metrics = accuracy_agg.merge(time_agg, on='Method', how='outer').merge(memory_agg, on='Method', how='outer')

# Include Data_Amount in aggregate_metrics
# Since Data_Amount per method is constant, we can just take one value from metrics_df
data_amount_map = metrics_df[['Method', 'Data_Amount']].drop_duplicates().set_index('Method')['Data_Amount'].to_dict()
aggregate_metrics['Data_Amount'] = aggregate_metrics['Method'].map(data_amount_map)

# Display the aggregated metrics
print("\nAggregated Metrics DataFrame (First 5 Rows):")
print(aggregate_metrics.head())

# Check for zeros in Average_Time_Sec and Average_Memory_MB
zero_time = (aggregate_metrics['Average_Time_Sec'] == 0)
zero_memory = (aggregate_metrics['Average_Memory_MB'] == 0)
if zero_time.any():
    print("\nWarning: Some methods have 'Average_Time_Sec' equal to zero.")
    print(aggregate_metrics[zero_time][['Method', 'Average_Time_Sec']])
if zero_memory.any():
    print("\nWarning: Some methods have 'Average_Memory_MB' equal to zero.")
    print(aggregate_metrics[zero_memory][['Method', 'Average_Memory_MB']])

# Methods with data amounts for specialized plotting
methods_with_data_amount = aggregate_metrics[aggregate_metrics['Data_Amount'].notnull()]

# ---------------------------
# Graph 1: Accuracy Comparison
# ---------------------------
plt.figure(figsize=(12, 6))
sns.barplot(x='Method', y='Average_Accuracy', data=aggregate_metrics, palette='viridis')
plt.title('Average Accuracy Comparison of Methods')
plt.xlabel('Method')
plt.ylabel('Average Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
plt.show()

# ---------------------------
# Graph 2: Accuracy per Bucket Comparison
# ---------------------------
# Check if 'Bucket' column exists
if 'Bucket' in metrics_df.columns:
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='Bucket', y='Accuracy', hue='Method', data=accuracy_df, marker='o')
    plt.title('Accuracy per Bucket for Each Method')
    plt.xlabel('Bucket')
    plt.ylabel('Accuracy')
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_per_bucket_comparison.png'))
    plt.show()
else:
    print("\nColumn 'Bucket' not found in metrics. Skipping Graph 2.")

# ---------------------------
# Graph 3: Dual-Axis Plot (Average Accuracy and Average Time)
# ---------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Method')
ax1.set_ylabel('Average Accuracy', color=color)
sns.barplot(x='Method', y='Average_Accuracy', data=aggregate_metrics, palette='Blues_d', ax=ax1)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(aggregate_metrics['Method'], rotation=45, ha='right')

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

color = 'tab:red'
sns.lineplot(x='Method', y='Average_Time_Sec', data=aggregate_metrics, marker='o', color=color, ax=ax2)
ax2.set_ylabel('Average Time (Sec)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Average Accuracy and Time Comparison of Methods')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_time_dual_axis.png'))
plt.show()

# ---------------------------
# Graph 4: Dual-Axis Plot (Average Accuracy and Average Memory)
# ---------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:green'
ax1.set_xlabel('Method')
ax1.set_ylabel('Average Accuracy', color=color)
sns.barplot(x='Method', y='Average_Accuracy', data=aggregate_metrics, palette='Greens_d', ax=ax1)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(aggregate_metrics['Method'], rotation=45, ha='right')

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

color = 'tab:orange'
sns.lineplot(x='Method', y='Average_Memory_MB', data=aggregate_metrics, marker='o', color=color, ax=ax2)
ax2.set_ylabel('Average Memory Footprint (MB)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Average Accuracy and Memory Footprint Comparison of Methods')
fig.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_memory_dual_axis.png'))
plt.show()

# ---------------------------
# Graph 5: Accuracy vs. Memory Footprint for Varying Data Amounts
# ---------------------------
if not methods_with_data_amount.empty:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Average_Memory_MB',
        y='Average_Accuracy',
        hue='Data_Amount',
        size='Data_Amount',
        sizes=(100, 400),
        data=methods_with_data_amount,
        palette='coolwarm',
        legend='full'
    )
    plt.title('Accuracy vs. Memory Footprint for Varying Data Amounts')
    plt.xlabel('Average Memory Footprint (MB)')
    plt.ylabel('Average Accuracy')
    plt.legend(title='Data Amount', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_memory_varying_data.png'))
    plt.show()
else:
    print("\nNo methods with data amount parameters found. Skipping Graph 5.")

# ---------------------------
# Save the aggregated metrics to a CSV
# ---------------------------
aggregate_metrics.to_csv(os.path.join(output_dir, 'aggregated_metrics.csv'), index=False)
print(f"\nAggregated metrics saved to '{os.path.join(output_dir, 'aggregated_metrics.csv')}'.")

print(f"\nAll graphs have been saved in the '{output_dir}' directory.")
