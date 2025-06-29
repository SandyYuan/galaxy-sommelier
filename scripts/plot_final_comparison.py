import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_per_task_correlation():
    """
    Generates and saves a bar plot comparing the per-task correlation
    for the SDSS-only, Mixed, and Max Overlap models.
    """
    output_dir = Path("benchmark_results/comparison_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data from the latest comparison runs with corrected max overlap model (epoch 3)
    data = {
        'Feature': [
            't03_bar_a06_bar_fraction',
            't04_spiral_a09_no_spiral_fraction', 
            't03_bar_a07_no_bar_fraction',
            't04_spiral_a08_spiral_fraction',
            't01_smooth_or_features_a03_star_or_artifact_fraction',
            't01_smooth_or_features_a02_features_or_disk_fraction',
            't01_smooth_or_features_a01_smooth_fraction',
            't02_edgeon_a04_yes_fraction',
            't02_edgeon_a05_no_fraction'
        ],
        'SDSS-only Correlation': [0.504843, 0.471730, 0.357965, 0.663404, 0.471244, 0.830521, 0.805700, 0.748225, 0.658756],
        'Mixed Model Correlation': [0.578755, 0.543781, 0.425297, 0.698296, 0.519300, 0.774724, 0.726697, 0.795632, 0.693946],
        'Max Overlap Correlation': [0.440979, 0.340584, 0.238200, 0.590113, 0.302107, 0.753756, 0.667092, 0.712984, 0.542849]
    }
    df = pd.DataFrame(data)

    # Create more readable names for the plot
    feature_map = {
        't03_bar_a06_bar_fraction': 'Bar',
        't04_spiral_a09_no_spiral_fraction': 'No Spiral',
        't03_bar_a07_no_bar_fraction': 'No Bar', 
        't04_spiral_a08_spiral_fraction': 'Spiral',
        't01_smooth_or_features_a03_star_or_artifact_fraction': 'Star/Artifact',
        't01_smooth_or_features_a02_features_or_disk_fraction': 'Features/Disk',
        't01_smooth_or_features_a01_smooth_fraction': 'Smooth',
        't02_edgeon_a04_yes_fraction': 'Edge-on',
        't02_edgeon_a05_no_fraction': 'Not Edge-on'
    }
    df['Feature'] = df['Feature'].map(feature_map)

    # Sort by the mixed model performance for consistent ordering
    df = df.sort_values(by='Mixed Model Correlation', ascending=True)

    # Melt the dataframe for easy plotting with seaborn
    df_melted = df.melt(id_vars='Feature', 
                        value_vars=['SDSS-only Correlation', 'Mixed Model Correlation', 'Max Overlap Correlation'], 
                        var_name='Model', value_name='Correlation')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))

    # Use a distinct color palette for three models
    colors = ['#2E8B57', '#FF6347', '#4169E1']  # Sea green, tomato, royal blue
    sns.barplot(data=df_melted, x='Correlation', y='Feature', hue='Model', 
                ax=ax, palette=colors, orient='h')

    ax.set_title('Three-Model Comparison: Per-Task Correlation on UKIDSS Data', 
                 fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Pearson Correlation', fontsize=14)
    ax.set_ylabel('Morphological Feature', fontsize=14)
    ax.legend(title='Model', loc='lower right', fontsize=12, title_fontsize=12)
    ax.set_xlim(0, 1)

    # Add value labels to the bars
    for p in ax.patches:
        width = p.get_width()
        if width > 0.05:  # Only add labels if bar is wide enough
            ax.text(width + 0.01, p.get_y() + p.get_height() / 2,
                    f'{width:.3f}',
                    va='center', fontsize=9)

    plt.tight_layout()

    plot_path = output_dir / "three_model_per_task_correlation_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {plot_path}")

def plot_overall_metrics_comparison():
    """
    Creates a summary plot comparing overall metrics across all three models.
    """
    output_dir = Path("benchmark_results/comparison_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall metrics from latest comparison
    metrics_data = {
        'Model': ['SDSS-only', 'Mixed Model', 'Max Overlap'],
        'Overall Correlation': [0.779, 0.810, 0.796],  # Approximate values from previous runs
        'R-squared': [0.607, 0.656, 0.633],
        'MAE': [0.165, 0.158, 0.159],
        'MSE': [0.057, 0.051, 0.056]
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Overall Performance Comparison: SDSS-only vs Mixed vs Max Overlap', 
                 fontsize=16, weight='bold')
    
    colors = ['#2E8B57', '#FF6347', '#4169E1']
    
    # Correlation
    sns.barplot(data=df, x='Model', y='Overall Correlation', ax=axes[0,0], palette=colors)
    axes[0,0].set_title('Overall Correlation')
    axes[0,0].set_ylim(0.7, 0.82)
    for i, v in enumerate(df['Overall Correlation']):
        axes[0,0].text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom')
    
    # R-squared
    sns.barplot(data=df, x='Model', y='R-squared', ax=axes[0,1], palette=colors)
    axes[0,1].set_title('R-squared')
    axes[0,1].set_ylim(0.6, 0.67)
    for i, v in enumerate(df['R-squared']):
        axes[0,1].text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom')
    
    # MAE (lower is better)
    sns.barplot(data=df, x='Model', y='MAE', ax=axes[1,0], palette=colors)
    axes[1,0].set_title('Mean Absolute Error (lower is better)')
    axes[1,0].set_ylim(0.155, 0.167)
    for i, v in enumerate(df['MAE']):
        axes[1,0].text(i, v + 0.0005, f'{v:.3f}', ha='center', va='bottom')
    
    # MSE (lower is better)
    sns.barplot(data=df, x='Model', y='MSE', ax=axes[1,1], palette=colors)
    axes[1,1].set_title('Mean Squared Error (lower is better)')
    axes[1,1].set_ylim(0.05, 0.058)
    for i, v in enumerate(df['MSE']):
        axes[1,1].text(i, v + 0.0003, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    plot_path = output_dir / "three_model_overall_metrics_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"Overall metrics plot saved to {plot_path}")

if __name__ == "__main__":
    plot_per_task_correlation()
    plot_overall_metrics_comparison()
    print("\nAll comparison plots generated successfully!") 