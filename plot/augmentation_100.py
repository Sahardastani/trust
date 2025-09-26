import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Sorted data
methods_sorted = ['Jitter', 'Crop', 'Source only', 'Rotation', 'Ours']
means_sorted = [39.9, 41.0, 41.2, 42.1, 54.3]

colors = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5']


# Plotting
plt.figure(figsize=(9, 6))
bars = plt.bar(methods_sorted, means_sorted, color=colors)
plt.ylabel("Accuracy %", fontsize=18)
plt.title("Accuracy vs. Augmentation Methods on CIFAR-100-C", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(35, max(means_sorted) + 2)  # Start y-axis from 30

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}",
             ha='center', va='bottom', fontsize=16)

plt.tight_layout()
plt.grid(True, axis='y', linestyle=':', linewidth=1, alpha=0.7)
plt.show()
plt.savefig('images/augmentation_100.png')
plt.savefig('pdfs/augmentation_100.pdf')
