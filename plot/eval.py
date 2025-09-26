import matplotlib.pyplot as plt

# Data
permutations = ['abcd', 'abdc', 'adcb', 'bacd', 'badc', 'dbca']
mean_scores = [77.5, 75.3, 74.2, 74.9, 71.6, 75.7]

fig, ax = plt.subplots(figsize=(10.5, 6.5))

# Plot data
ax.plot(permutations, mean_scores, marker='o', linestyle='-', linewidth=2, markersize=8, color='orange')

# Axis labels
ax.set_xlabel('Traversal Permutation in Evaluation', fontsize=26, labelpad=10)
ax.set_ylabel('Accuracy (%)', fontsize=26, labelpad=10)

# Axis limits
ax.set_xlim(-0.5, len(permutations) - 0.5)
ax.set_ylim(70, 80)

# Ticks and font sizes
ax.set_xticks(range(len(permutations)))
ax.set_xticklabels(permutations, fontsize=24)
ax.tick_params(axis='y', labelsize=24)

# Annotate points
for i, val in enumerate(mean_scores):
    ax.text(i, val + 0.3, f'{val:.1f}', ha='center', va='bottom', fontsize=22)

# Add arrowheads to axis ends
# x-axis arrow
ax.annotate('', xy=(len(permutations) - 0.45, 70), xytext=(len(permutations) - 0.6, 70),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)

# y-axis arrow
# New y-axis arrow (shifted left)
ax.annotate('', xy=(-0.5, 81), xytext=(-0.5, 79.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)

# Keep spines visible
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.savefig("images/eval.png")
plt.savefig("pdfs/eval.pdf")
plt.show()
