import matplotlib.pyplot as plt
import numpy as np

# Data
variations = [2.0, 4.0, 6.0]
cifar10_c = [75.6, 77.1, 77.5]
cifar100_c = [51.7, 53.7, 54.3]
imagenet_c = [54.4, 55.6, 56.1]

bar_width = 0.25
x = np.arange(len(variations))  # base x positions

color_cifar10 = '#1f77b4'
color_cifar100 = '#ff7f0e'
color_imagenet = '#2ca02c'

# Plot
fig, ax = plt.subplots(figsize=(10.5, 6.5))
bars1 = ax.bar(x - bar_width, cifar10_c, width=bar_width, label='CIFAR10-C', color=color_cifar10)
bars2 = ax.bar(x, cifar100_c, width=bar_width, label='CIFAR100-C', color=color_cifar100)
bars3 = ax.bar(x + bar_width, imagenet_c, width=bar_width, label='ImageNet-C', color=color_imagenet)

# Add text labels on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.1f}', 
                ha='center', va='bottom', fontsize=22)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# Axis labels and ticks
ax.set_xlabel('Traversal Permutation Number', fontsize=26, labelpad=10)
ax.set_ylabel('Accuracy (%)', fontsize=26, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels([str(v) for v in variations], fontsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.set_ylim(50, 80)

# Spines and grid
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.grid(True, axis='y', linestyle=':', alpha=0.7)

# Add axis-end arrows
# X-axis arrow (to the right of last group)
ax.annotate('', xy=(2.6, 50), xytext=(2.5, 50),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)

# Y-axis arrow (slightly left of y-axis)
ax.annotate('', xy=(-0.51, 81), xytext=(-0.51, 79.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)

# Legend
ax.legend(fontsize=22, loc='center right')

# Save and show
plt.tight_layout()
plt.savefig("images/variation.png")
plt.savefig("pdfs/variation.pdf")
plt.show()
