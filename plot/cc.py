import matplotlib.pyplot as plt
import numpy as np

# Data
traversals = [1, 2, 4, 6, 8]
memory = [0.998, 1.178, 1.580, 1.954, 2.36]
baseline = memory[0]
colors = ['#1f77b4'] + ['gray'] * (len(traversals) - 1)

# Create the plot
fig, ax = plt.subplots(figsize=(10.5, 6.5))
bars = ax.bar(traversals, memory, color=colors)

# Annotate bars
for i, val in enumerate(memory):
    if i == 0:
        ax.text(traversals[i], val + 0.05, f"{val:.3f}", ha='center', va='bottom', fontsize=22)
    else:
        ax.text(traversals[i], val + 0.05, f"x{val / baseline:.2f}", ha='center', va='bottom', fontsize=22)

# Axis settings
ax.set_xlim(0, max(traversals) + 1)
ax.set_ylim(0, max(memory) + 0.4)
ax.set_xticks(traversals)
ax.set_xlabel("Number of Traversals", fontsize=26)
ax.set_ylabel("GPU Memory (GB)", fontsize=26)
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
# ax.set_title("GPU Memory Usage vs Traversal Count", fontsize=22)

# Remove all spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Add axis arrows
# Update y-axis start from 0 to 0.5
ax.set_ylim(0.5, max(memory) + 0.4)

# Update custom axis arrows accordingly
ax.plot([0, max(traversals) + 0.6], [0.5, 0.5], color='black', clip_on=False)
ax.plot([0, 0], [0.5, max(memory) + 0.1], color='black', clip_on=False)
ax.annotate('', xy=(max(traversals) + 0.8, 0.5), xytext=(0, 0.5),
            arrowprops=dict(arrowstyle='->', linewidth=1.5, color='black'))
ax.annotate('', xy=(0, max(memory) + 0.3), xytext=(0, 0.5),
            arrowprops=dict(arrowstyle='->', linewidth=1.5, color='black'))


# Save and show
plt.tight_layout()
plt.savefig("images/cc.png", dpi=300)
plt.savefig("pdfs/cc.pdf", dpi=300)
plt.show()
