import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_per_task_correlation():
    """
    Generates and saves a bar plot comparing the per-task correlation
    for the SDSS-only and Mixed models.
    """
    output_dir = Path("benchmark_results/comparison_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data from the final full-dataset run
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
        'Mixed Model Correlation': [0.589888, 0.554826, 0.426390, 0.729861, 0.534782, 0.893810, 0.864938, 0.800587, 0.698937]
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

    # Sort by the performance improvement
    df['Difference'] = df['Mixed Model Correlation'] - df['SDSS-only Correlation']
    df = df.sort_values(by='Difference', ascending=False)

    # Melt the dataframe for easy plotting with seaborn
    df_melted = df.melt(id_vars='Feature', value_vars=['SDSS-only Correlation', 'Mixed Model Correlation'], 
                        var_name='Model', value_name='Correlation')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.barplot(data=df_melted, x='Correlation', y='Feature', hue='Model', ax=ax, palette='husl', orient='h')

    ax.set_title('Per-Task Correlation Comparison on UKIDSS Data', fontsize=16, weight='bold')
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_ylabel('Morphological Feature', fontsize=12)
    ax.legend(title='Model', loc='lower right')
    ax.set_xlim(0, 1)

    # Add value labels to the bars
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.01, p.get_y() + p.get_height() / 2,
                f'{width:.3f}',
                va='center')

    plt.tight_layout()

    plot_path = output_dir / "per_task_correlation_comparison.png"
    plt.savefig(plot_path, dpi=300)
    
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_per_task_correlation() 