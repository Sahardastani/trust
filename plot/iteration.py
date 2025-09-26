import matplotlib.pyplot as plt

# Data
iteration = [1, 2, 4, 6, 8]
trust_acc = [74.2, 76.26, 77.22, 77.35, 76.24]
tent_acc = [66.53, 67.18, 68.16, 68.88, 69.36]

# Create the plot
fig, ax = plt.subplots(figsize=(10.5, 6.5))
ax.plot(iteration, trust_acc, marker='o', label='TRUST', linewidth=2)
ax.plot(iteration, tent_acc, marker='s', label='Tent', linewidth=2)

# Axis labels
ax.set_xlabel('Batch Size', fontsize=26, labelpad=10)
ax.set_ylabel('Accuracy (%)', fontsize=26, labelpad=10)

# Tick settings
ax.set_xticks(iteration)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)

# Axis limits
ax.set_xlim(0.5, 8.8)
ax.set_ylim(66, 78)

# Axis spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Arrows for axes
ax.annotate('', xy=(8.9, 66), xytext=(8.6, 66),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)
ax.annotate('', xy=(0.5, 78.5), xytext=(0.5, 78.1),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)

# Grid and legend
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(fontsize=22)

# Save the plot
plt.tight_layout()
plt.savefig("images/iteration.png", dpi=300)
plt.savefig("pdfs/iteration.pdf", dpi=300)
plt.show()
