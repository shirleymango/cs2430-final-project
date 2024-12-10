# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load metrics from CSV
# df_metrics = pd.read_csv('continual_learning_metrics.csv')

# # Plot Time Taken per Operation
# plt.figure(figsize=(12, 6))
# sns.barplot(x='Operation', y='Time_Sec', hue='Bucket', data=df_metrics)
# plt.title('Time Taken per Operation per Bucket')
# plt.xlabel('Operation')
# plt.ylabel('Time (seconds)')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # Plot Memory Usage per Operation
# plt.figure(figsize=(12, 6))
# sns.barplot(x='Operation', y='Memory_MB', hue='Bucket', data=df_metrics)
# plt.title('Memory Change per Operation per Bucket')
# plt.xlabel('Operation')
# plt.ylabel('Memory Change (MB)')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # Plot CPU Usage per Operation
# plt.figure(figsize=(12, 6))
# sns.barplot(x='Operation', y='CPU_Usage_Percent', hue='Bucket', data=df_metrics)
# plt.title('CPU Usage per Operation per Bucket')
# plt.xlabel('Operation')
# plt.ylabel('CPU Usage (%)')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Path to the metrics CSV file
metrics_file = 'continual_learning_metrics.csv'

# Check if the metrics file exists
if not os.path.exists(metrics_file):
    print(f"Metrics file '{metrics_file}' not found in the current directory.")
    exit(1)

# Read the metrics CSV into a pandas DataFrame
df = pd.read_csv(metrics_file)

# Function to extract numerical bucket number from bucket name
def extract_bucket_num(bucket_name):
    # Assumes bucket names are in the format 'bucket_1.arff', 'bucket_2.arff', etc.
    base = os.path.splitext(bucket_name)[0]  # Removes '.arff'
    try:
        num = int(base.split('_')[1])
    except (IndexError, ValueError):
        num = 0
    return num

# Apply the function to create a new column 'Bucket_Num'
df['Bucket_Num'] = df['Bucket'].apply(extract_bucket_num)

# Sort the DataFrame based on 'Bucket_Num' to ensure chronological order
df = df.sort_values('Bucket_Num')

# Define the order of operations for consistent coloring in plots
operation_order = ['Load', 'Scaling', 'Training', 'Prediction']

# ============================
# Plot 1: Time Taken per Operation per Bucket
# ============================
plt.figure(figsize=(14, 8))
sns.barplot(
    x='Bucket_Num',
    y='Time_Sec',
    hue='Operation',
    data=df,
    order=sorted(df['Bucket_Num'].unique()),
    hue_order=operation_order
)
plt.title('Time Taken per Operation per Bucket', fontsize=16)
plt.xlabel('Bucket Number', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.legend(title='Operation', fontsize=12, title_fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('time_per_operation_per_bucket.png')
plt.show()

# ============================
# Plot 2: Memory Change per Operation per Bucket
# ============================
plt.figure(figsize=(14, 8))
sns.barplot(
    x='Bucket_Num',
    y='Memory_MB',
    hue='Operation',
    data=df,
    order=sorted(df['Bucket_Num'].unique()),
    hue_order=operation_order
)
plt.title('Memory Change per Operation per Bucket', fontsize=16)
plt.xlabel('Bucket Number', fontsize=14)
plt.ylabel('Memory Change (MB)', fontsize=14)
plt.legend(title='Operation', fontsize=12, title_fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('memory_per_operation_per_bucket.png')
plt.show()

# ============================
# Plot 3: CPU Usage per Operation per Bucket
# ============================
plt.figure(figsize=(14, 8))
sns.barplot(
    x='Bucket_Num',
    y='CPU_Usage_Percent',
    hue='Operation',
    data=df,
    order=sorted(df['Bucket_Num'].unique()),
    hue_order=operation_order
)
plt.title('CPU Usage per Operation per Bucket', fontsize=16)
plt.xlabel('Bucket Number', fontsize=14)
plt.ylabel('CPU Usage (%)', fontsize=14)
plt.legend(title='Operation', fontsize=12, title_fontsize=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('cpu_usage_per_operation_per_bucket.png')
plt.show()

# ============================
# Plot 4: Total Time per Bucket by Operation (Stacked Bar)
# ============================
# Pivot the DataFrame to have operations as columns
time_pivot = df.pivot_table(
    index='Bucket_Num',
    columns='Operation',
    values='Time_Sec',
    aggfunc='sum'
).fillna(0)

# Plot stacked bar chart
time_pivot.plot(
    kind='bar',
    stacked=True,
    figsize=(14, 8),
    colormap='tab20'
)
plt.title('Total Time per Bucket by Operation', fontsize=16)
plt.xlabel('Bucket Number', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.legend(title='Operation', fontsize=12, title_fontsize=13)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('total_time_per_bucket_stacked.png')
plt.show()

# ============================
# Plot 5: Total Memory Change per Bucket by Operation (Stacked Bar)
# ============================
# Pivot the DataFrame to have operations as columns
memory_pivot = df.pivot_table(
    index='Bucket_Num',
    columns='Operation',
    values='Memory_MB',
    aggfunc='sum'
).fillna(0)

# Plot stacked bar chart
memory_pivot.plot(
    kind='bar',
    stacked=True,
    figsize=(14, 8),
    colormap='tab20'
)
plt.title('Total Memory Change per Bucket by Operation', fontsize=16)
plt.xlabel('Bucket Number', fontsize=14)
plt.ylabel('Memory Change (MB)', fontsize=14)
plt.legend(title='Operation', fontsize=12, title_fontsize=13)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('total_memory_per_bucket_stacked.png')
plt.show()

# ============================
# Plot 6: Distribution of Time Taken Across Operations
# ============================
plt.figure(figsize=(14, 8))
sns.boxplot(
    x='Operation',
    y='Time_Sec',
    data=df,
    order=operation_order,
    palette='Set2'
)
plt.title('Distribution of Time Taken per Operation', fontsize=16)
plt.xlabel('Operation', fontsize=14)
plt.ylabel('Time (seconds)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('time_distribution_per_operation.png')
plt.show()

# ============================
# Plot 7: Distribution of Memory Change Across Operations
# ============================
plt.figure(figsize=(14, 8))
sns.boxplot(
    x='Operation',
    y='Memory_MB',
    data=df,
    order=operation_order,
    palette='Set3'
)
plt.title('Distribution of Memory Change per Operation', fontsize=16)
plt.xlabel('Operation', fontsize=14)
plt.ylabel('Memory Change (MB)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('memory_distribution_per_operation.png')
plt.show()

# ============================
# Plot 8: Distribution of CPU Usage Across Operations
# ============================
plt.figure(figsize=(14, 8))
sns.boxplot(
    x='Operation',
    y='CPU_Usage_Percent',
    data=df,
    order=operation_order,
    palette='Pastel1'
)
plt.title('Distribution of CPU Usage per Operation', fontsize=16)
plt.xlabel('Operation', fontsize=14)
plt.ylabel('CPU Usage (%)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('cpu_usage_distribution_per_operation.png')
plt.show()

# ============================
# Optional: Plot Peak Memory Usage and Total Learning Time
# ============================
# Assuming that peak memory usage and total time were printed but not saved in the CSV,
# you might want to manually add them or modify the main script to save them.

# For demonstration, let's assume you have these values:
# (Replace these with actual values from your main script's output)
total_learning_time = 120.50  # seconds
peak_memory_usage = 200.75     # MB

# Create a simple bar chart for total time and peak memory
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar for Total Time
sns.barplot(
    x=['Total Continual Learning'],
    y=[total_learning_time],
    color='skyblue',
    ax=ax1
)
ax1.set_ylabel('Time (seconds)', fontsize=14)
ax1.set_title('Overall Continual Learning Metrics', fontsize=16)

# Create a second y-axis for Peak Memory Usage
ax2 = ax1.twinx()
sns.barplot(
    x=['Total Continual Learning'],
    y=[peak_memory_usage],
    color='salmon',
    ax=ax2
)
ax2.set_ylabel('Peak Memory Usage (MB)', fontsize=14)

plt.tight_layout()
plt.savefig('overall_continual_learning_metrics.png')
plt.show()

# ============================
# Completion Message
# ============================
print("All plots have been generated and saved as PNG files in the current directory.")
