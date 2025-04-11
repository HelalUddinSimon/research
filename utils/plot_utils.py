import matplotlib.pyplot as plt
import numpy as np

def plot_flops_comparison(layer_flops_before, layer_flops_after, layer_names):
    plt.figure(figsize=(22, 12))
    bar_width = 0.35
    index = np.arange(len(layer_flops_before))
    plt.bar(index, layer_flops_before, bar_width, label='Before Pruning')
    plt.bar(index + bar_width, layer_flops_after, bar_width, label='After Pruning')
    plt.xlabel('Layers')
    plt.ylabel('FLOPS')
    plt.title('FLOPS per Layer Before and After Pruning')
    plt.xticks(index + bar_width / 2, layer_names, rotation=90)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def plot_metrics_comparison(phases, losses, accuracies):
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.bar(phases, losses, color=['blue', 'red', 'green'])
    plt.ylabel('Loss')
    plt.title('Model Loss at Different Phases')

    plt.subplot(1, 2, 2)
    plt.bar(phases, [acc * 100 for acc in accuracies], color=['blue', 'red', 'green'])
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy at Different Phases')
    return plt.gcf()

def plot_eigenvector_distribution(eigenvectors):
    magnitudes = [np.linalg.norm(vec) for vec in eigenvectors]
    plt.figure(figsize=(10, 5))
    plt.hist(magnitudes, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Eigenvector Magnitudes')
    plt.xlabel('Magnitude')
    plt.ylabel('Frequency')
    plt.grid(True)
    return plt.gcf()
