# plot_reduction_results.py

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy_memory(summary_df, output_dir, output_filename):
    """
    Generate a dual-axis plot with Sampling Multiplier on x-axis,
    Average Accuracy on left y-axis (bar plot),
    and Peak Memory on right y-axis (line plot).
    
    Parameters:
        summary_df (pd.DataFrame): DataFrame containing summary metrics.
        output_dir (str): Directory to save the plot.
        output_filename (str): Name of the output plot file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert Sampling_Multiplier to float if it's not already
    summary_df['Sampling_Multiplier'] = summary_df['Sampling_Multiplier'].astype(float)
    
    # Sort the DataFrame by Sampling_Multiplier for better visualization
    summary_df = summary_df.sort_values('Sampling_Multiplier')
    
    # Initialize the matplotlib figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Set seaborn style for aesthetics
    sns.set(style="whitegrid")
    
    # Plot Average Accuracy as a bar plot on ax1
    bars = sns.barplot(
        x='Sampling_Multiplier',
        y='Average_Accuracy',
        data=summary_df,
        color='skyblue',
        ax=ax1,
        label='Average Accuracy'
    )
    
    # Set labels and title for the first y-axis
    ax1.set_xlabel('Sampling Multiplier', fontsize=14)
    ax1.set_ylabel('Average Accuracy', fontsize=14, color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_ylim(0, 1)  # Assuming accuracy ranges between 0 and 1
    
    # Create a second y-axis for Peak Memory
    ax2 = ax1.twinx()
    
    # Align x-ticks for Peak Memory plot with bar plot
    x_positions = [bar.get_x() + bar.get_width() / 2.0 for bar in bars.patches]
    sns.lineplot(
        x=x_positions,
        y=summary_df['Peak_Memory_MB'],
        color='darkorange',
        marker='o',
        linewidth=2,
        ax=ax2,
        label='Peak Memory (MB)'
    )
    
    # Set labels for the second y-axis
    ax2.set_ylabel('Peak Memory Usage (MB)', fontsize=14, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    # Optionally, set limits for the memory axis for better visualization
    memory_min = summary_df['Peak_Memory_MB'].min() * 0.95
    memory_max = summary_df['Peak_Memory_MB'].max() * 1.05
    ax2.set_ylim(memory_min, memory_max)
    
    # Add a title to the plot
    plt.title('Average Accuracy and Peak Memory Usage vs Sampling Multiplier', fontsize=16)
    
    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='upper left', fontsize=12)
    
    # Improve layout to prevent overlap
    fig.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, output_filename)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved to {plot_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot Average Accuracy and Peak Memory vs Sampling Multiplier")
    parser.add_argument('--summary_csv', type=str, required=True, help='Path to the summary metrics CSV file')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save the plot')
    parser.add_argument('--output_filename', type=str, default='accuracy_memory_vs_sampling_multiplier.png', help='Filename for the output plot')
    
    args = parser.parse_args()
    
    # Check if the summary CSV exists
    if not os.path.exists(args.summary_csv):
        print(f"Error: Summary CSV file not found at {args.summary_csv}")
        return
    
    # Load the summary metrics
    try:
        summary_df = pd.read_csv(args.summary_csv)
    except Exception as e:
        print(f"Error reading the summary CSV file: {e}")
        return
    
    # Check if required columns exist
    required_columns = {'Sampling_Multiplier', 'Average_Accuracy', 'Peak_Memory_MB'}
    if not required_columns.issubset(summary_df.columns):
        print(f"Error: The summary CSV must contain the following columns: {required_columns}")
        return
    
    # Plot the data
    plot_accuracy_memory(summary_df, args.output_dir, args.output_filename)

if __name__ == "__main__":
    main()
