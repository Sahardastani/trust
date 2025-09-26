import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# Sorted data
methods_sorted = ['Source', 'Rotation', 'Crop', 'Jitter', 'TRUST']
means_sorted = [65.9, 66.8, 66.9, 68.3, 77.5]

colors = ['#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5']

# Plotting
fig, ax = plt.subplots(figsize=(10.5, 6.5))
bars = ax.bar(methods_sorted, means_sorted, color=colors)

# Labels
ax.set_ylabel("Accuracy %", fontsize=26, labelpad=10)
ax.set_ylim(60, max(means_sorted) + 2)
ax.set_xticks(range(len(methods_sorted)))
ax.set_xticklabels(methods_sorted, fontsize=24)
ax.tick_params(axis='y', labelsize=24)

# Annotate bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}",
            ha='center', va='bottom', fontsize=22)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Axis-end arrows
# x-axis arrow (after the last bar)
ax.annotate('', xy=(4.7, 60), xytext=(4.6, 60),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)

# y-axis arrow (slightly left of axis)
ax.annotate('', xy=(-0.64, 80), xytext=(-0.64, 79),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)

# Grid and layout
ax.grid(True, axis='y', linestyle=':', linewidth=1, alpha=0.7)
plt.tight_layout()

# Save and show
plt.savefig('images/augmentation_10.png', bbox_inches='tight')
plt.savefig('pdfs/augmentation_10.pdf', bbox_inches='tight')

plt.show()
