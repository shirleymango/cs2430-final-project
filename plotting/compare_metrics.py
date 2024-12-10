import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Configuration Parameters
# -----------------------------

# Define the root directory containing metrics for all methods
# Adjust this path based on your actual directory structure
metrics_root_dir = '../continual_learning/'

# Define the list of methods to include in the comparison
# Ensure that the baseline method is included in this list
method_names = ['Baseline', 'Method_A', 'Method_B', 'Method_C']  # Replace with your actual method names

# Define the name of the baseline method
baseline_method = 'Baseline'  # Replace with your actual baseline method name

# Define the output directory for plots
output_dir = '../plotting/plots'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory set to: {output_dir}")

# -----------------------------
# Function Definitions
# -----------------------------

def extract_bucket_num(bucket_name):
    """
    Extract numerical bucket number from bucket filename.
    Assumes bucket names are in the format 'bucket_1.arff', 'bucket_2.arff', etc.
    """
    base = os.path.splitext(bucket_name)[0]  # Removes '.arff'
    try:
        num = int(base.split('_')[1])
    except (IndexError, ValueError):
        num = 0
    return num

def load_metrics(metrics_root, methods):
    """
    Load metrics from specified method directories.
    
    Parameters:
        metrics_root (str): Path to the root directory containing method subdirectories.
        methods (list): List of method names to include.
    
    Returns:
        pd.DataFrame: Combined dataframe containing metrics from specified methods.
    """
    methods_data = []
    print("\nStarting to load metrics for specified methods...\n")
    
    for method_name in methods:
        method_dir = os.path.join(metrics_root, method_name)
        metrics_file = os.path.join(method_dir, 'continual_learning_metrics.csv')
        print(f"Loading metrics for method: {method_name}")
        print(f"Expected metrics file at: {metrics_file}")
        
        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                df['Method'] = method_name
                methods_data.append(df)
                print(f"Successfully loaded metrics for method: {method_name}")
                print(f"Number of entries loaded: {len(df)}\n")
            except Exception as e:
                print(f"Error loading metrics for method '{method_name}': {e}\n")
        else:
            print(f"Warning: Metrics file not found for method '{method_name}' at '{metrics_file}'. Skipping.\n")
    
    if not methods_data:
        raise FileNotFoundError("No metrics data found for the specified methods. Please check your directory structure and method names.")
    
    combined_df = pd.concat(methods_data, ignore_index=True)
    print("All specified methods' metrics have been loaded and combined.\n")
    
    # Extract bucket numbers for sorting
    combined_df['Bucket_Num'] = combined_df['Bucket'].apply(extract_bucket_num)
    
    # Sort the DataFrame based on 'Method' and 'Bucket_Num' to ensure chronological order
    combined_df = combined_df.sort_values(['Method', 'Bucket_Num'])
    print("Combined dataframe sorted by Method and Bucket Number.\n")
    
    print(f"Total entries in combined dataframe: {len(combined_df)}\n")
    
    return combined_df

def calculate_average_metrics(df):
    """
    Calculate average metrics per method and operation.
    
    Parameters:
        df (pd.DataFrame): Combined dataframe containing metrics from all methods.
    
    Returns:
        pd.DataFrame: DataFrame containing average metrics.
    """
    print("Calculating average metrics per method and operation...\n")
    # Define the operations to include
    operations = ['Load', 'Scaling', 'Training', 'Prediction', 'Accuracy']
    
    # Filter relevant operations
    df_filtered = df[df['Operation'].isin(operations)]
    print(f"Number of entries after filtering operations: {len(df_filtered)}\n")
    
    # Group by Method and Operation, then calculate mean of Time_Sec, Memory_MB, CPU_Usage_Percent, and Accuracy
    avg_metrics = df_filtered.groupby(['Method', 'Operation']).agg({
        'Time_Sec': 'mean',
        'Memory_MB': 'mean',
        'CPU_Usage_Percent': 'mean',
        'Accuracy': 'mean'
    }).reset_index()
    
    print("Average metrics calculated successfully.\n")
    print(avg_metrics.head(), "\n")
    
    return avg_metrics

def calculate_summary_metrics(avg_metrics):
    """
    Calculate summary metrics per method:
    - Average Accuracy
    - Total Memory Footprint
    - Total Time
    
    Parameters:
        avg_metrics (pd.DataFrame): DataFrame containing average metrics.
    
    Returns:
        pd.DataFrame: DataFrame containing summary metrics per method.
    """
    print("Calculating summary metrics per method...\n")
    summary = avg_metrics.copy()
    
    # Pivot the dataframe to have operations as columns
    pivot = summary.pivot(index='Method', columns='Operation', values=['Time_Sec', 'Memory_MB', 'Accuracy'])
    
    # Flatten MultiIndex columns
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    
    # Calculate Total Time and Total Memory Footprint
    pivot['Total_Time_Sec'] = pivot['Time_Sec_Load'] + pivot['Time_Sec_Scaling'] + pivot['Time_Sec_Training'] + pivot['Time_Sec_Prediction']
    pivot['Total_Memory_MB'] = pivot['Memory_MB_Load'] + pivot['Memory_MB_Scaling'] + pivot['Memory_MB_Training'] + pivot['Memory_MB_Prediction']
    
    # Calculate Average Accuracy
    pivot['Average_Accuracy'] = pivot['Accuracy_Accuracy']
    
    # Reset index to make 'Method' a column again
    pivot = pivot.reset_index()
    
    # Select relevant columns
    summary_metrics = pivot[['Method', 'Average_Accuracy', 'Total_Memory_MB', 'Total_Time_Sec']]
    
    print("Summary metrics calculated successfully.\n")
    print(summary_metrics.head(), "\n")
    
    return summary_metrics

def plot_specific_metrics(summary_metrics, output_path):
    """
    Generate and save the specific plots:
    1. Average Accuracy across methods
    2. Average Total Memory Footprint across methods
    3. Average Time across methods
    4. Accuracy vs Time for each method (dual axis plot)
    
    Parameters:
        summary_metrics (pd.DataFrame): DataFrame containing summary metrics per method.
        output_path (str): Directory to save the plots.
    """
    print("Starting to generate specific plots...\n")
    
    # Set seaborn style
    sns.set(style="whitegrid")
    
    # -----------------------------
    # Plot 1: Average Accuracy across Methods
    # -----------------------------
    print("Generating Plot 1: Average Accuracy across Methods")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Method',
        y='Average_Accuracy',
        data=summary_metrics,
        palette='viridis'
    )
    plt.title('Average Accuracy across Methods', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Average Accuracy', fontsize=14)
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plot1_path = os.path.join(output_path, 'average_accuracy_across_methods.png')
    plt.savefig(plot1_path)
    plt.close()
    print(f"Plot 1 saved: {plot1_path}\n")
    
    # -----------------------------
    # Plot 2: Average Total Memory Footprint across Methods
    # -----------------------------
    print("Generating Plot 2: Average Total Memory Footprint across Methods")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Method',
        y='Total_Memory_MB',
        data=summary_metrics,
        palette='magma'
    )
    plt.title('Average Total Memory Footprint across Methods', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Total Memory Footprint (MB)', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plot2_path = os.path.join(output_path, 'average_total_memory_across_methods.png')
    plt.savefig(plot2_path)
    plt.close()
    print(f"Plot 2 saved: {plot2_path}\n")
    
    # -----------------------------
    # Plot 3: Average Time across Methods
    # -----------------------------
    print("Generating Plot 3: Average Time across Methods")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Method',
        y='Total_Time_Sec',
        data=summary_metrics,
        palette='plasma'
    )
    plt.title('Average Time across Methods', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Total Time (seconds)', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plot3_path = os.path.join(output_path, 'average_time_across_methods.png')
    plt.savefig(plot3_path)
    plt.close()
    print(f"Plot 3 saved: {plot3_path}\n")
    
    # -----------------------------
    # Plot 4: Accuracy vs Time for Each Method (Dual Axis Plot)
    # -----------------------------
    print("Generating Plot 4: Accuracy vs Time for Each Method (Dual Axis Plot)")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plot for Total Time
    sns.barplot(
        x='Method',
        y='Total_Time_Sec',
        data=summary_metrics,
        color='skyblue',
        ax=ax1,
        label='Total Time (sec)'
    )
    ax1.set_xlabel('Method', fontsize=14)
    ax1.set_ylabel('Total Time (seconds)', color='skyblue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_ylim(0, summary_metrics['Total_Time_Sec'].max() * 1.2)  # Add some space on top
    
    # Create a second y-axis for Average Accuracy
    ax2 = ax1.twinx()
    sns.lineplot(
        x='Method',
        y='Average_Accuracy',
        data=summary_metrics,
        color='darkorange',
        marker='o',
        ax=ax2,
        label='Average Accuracy'
    )
    ax2.set_ylabel('Average Accuracy', color='darkorange', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_ylim(0, 1)  # Accuracy ranges from 0 to 1
    
    # Add legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12)
    
    plt.title('Accuracy vs Time for Each Method', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plot4_path = os.path.join(output_path, 'accuracy_vs_time_across_methods.png')
    plt.savefig(plot4_path)
    plt.close()
    print(f"Plot 4 saved: {plot4_path}\n")
    
        # -----------------------------
        # Plot 5: Average Memory vs Average Accuracy Across Methods
        # -----------------------------
    print("Generating Plot 5: Average Memory vs Average Accuracy Across Methods")
        
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
            x='Total_Memory_MB',
            y='Average_Accuracy',
            data=summary_metrics,
            hue='Method',
            s=100,
            palette='deep',
            edgecolor='w'
        )
    plt.title('Average Memory vs Average Accuracy Across Methods', fontsize=16)
    plt.xlabel('Total Memory Footprint (MB)', fontsize=14)
    plt.ylabel('Average Accuracy', fontsize=14)
        
        # Annotate each point with the method name
    # for i in range(summary_metrics.shape[0]):
    #     plt.text(
    #             summary_metrics['Total_Memory_MB'].iloc[i] + 0.1,  # Slightly offset to the right
    #             summary_metrics['Average_Accuracy'].iloc[i],
    #             summary_metrics['Method'].iloc[i],
    #             horizontalalignment='left',
    #             size='medium',
    #             color='black',
    #             weight='semibold'
    #         )
        
    plt.legend(title='Method', fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plot5_path = os.path.join(output_path, 'average_memory_vs_average_accuracy_across_methods.png')
    plt.savefig(plot5_path)
    plt.close()
    print(f"Plot 5 saved: {plot5_path}\n")
    
    # -----------------------------
    # Completion Message
    # -----------------------------
    print("All specific plots have been generated and saved successfully.\n")

# -----------------------------
# Function Calls
# -----------------------------
if __name__ == "__main__":
    try:
        # Load metrics for the specified methods
        combined_df = load_metrics(metrics_root_dir, method_names)
        
        # Calculate average metrics
        avg_metrics = calculate_average_metrics(combined_df)
        
        # Calculate summary metrics
        summary_metrics = calculate_summary_metrics(avg_metrics)
        
        # Generate and save specific plots
        plot_specific_metrics(summary_metrics, output_dir)
        
    except Exception as e:
        print(f"An error occurred: {e}")
