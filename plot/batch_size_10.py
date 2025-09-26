import matplotlib.pyplot as plt

batch_sizes = [16, 32, 64, 128]

# CIFAR10
tent_values = [65.45, 66.3, 66.55, 66.53]
TRUST_values = [73.1, 77.02, 77.88, 77.52]

# CIFAR100 (commented out)
# tent_values = [40.77, 41.56, 41.79, 41.8]
# TRUST_values = [47.34, 52.53, 54.47, 54.3]

dataset = "CIFAR10"

# Plotting
fig, ax = plt.subplots(figsize=(10.5, 6.5))
ax.plot(batch_sizes, TRUST_values, marker='o', label='TRUST', linewidth=2)
ax.plot(batch_sizes, tent_values, marker='o', label='Tent', linewidth=2)

# Labels
ax.set_xlabel('Batch Size', fontsize=26, labelpad=10)
ax.set_ylabel('Accuracy (%)', fontsize=26, labelpad=10)

# Ticks
ax.set_xticks(batch_sizes)
ax.set_xticklabels(['16', '32', '64', '128'], fontsize=24)
ax.tick_params(axis='y', labelsize=24)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Thicken left and bottom spines
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Arrow for x-axis
ax.annotate('', xy=(135, 64.82), xytext=(128, 64.82),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)

# Arrow for y-axis (slightly left of axis)
ax.annotate('', xy=(10.4, 79), xytext=(10.4, 77.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'), annotation_clip=False)

# Grid and legend
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(fontsize=22)

# Layout and save
plt.tight_layout()
plt.savefig(f"images/batch_size_{dataset}.png")
plt.savefig(f"pdfs/batch_size_{dataset}.pdf")
plt.show()
