import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['Ensemble', 'Repetition', 'TRUST']
scores = [68.1, 69.6, 75.6]
colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

# Plot
fig, ax = plt.subplots(figsize=(10.5, 6.5))
bars = ax.bar(methods, scores, color=colors)

# Axis labels
ax.set_xlabel('Aggregation Methods', fontsize=26, labelpad=10)
ax.set_ylabel('Accuracy (%)', fontsize=26, labelpad=10)

# Axis ticks
ax.set_ylim(65, 78)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=24)
ax.tick_params(axis='y', labelsize=24)

# Annotate scores
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{score:.1f}',
            ha='center', va='bottom', fontsize=22)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Keep left and bottom spines thicker
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# === Add axis-end arrows ===
# x-axis arrow (right end)
ax.annotate('', xy=(2.6, 65), xytext=(2.5, 65),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)

# y-axis arrow (top end, slightly left)
ax.annotate('', xy=(-0.54, 78.5), xytext=(-0.54, 77.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)

# Grid and layout
ax.grid(True, axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("images/aggregation.png")
plt.savefig("pdfs/aggregation.pdf")
plt.show()
