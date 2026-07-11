import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_correlation_heatmap(df):
    plt.figure(figsize=(20, 16))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_scree(explained_variance_ratio):
    cumulative = np.cumsum(explained_variance_ratio)
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(cumulative) + 1),
        cumulative,
        marker='o'
    )
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Scree Plot')
    plt.legend()
    plt.savefig('scree_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    