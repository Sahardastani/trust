import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.patches as patches
# Ensure output folder exists
os.makedirs("images", exist_ok=True)

# Define permutations
traverses = [
    'abcd', 'abdc', 'acbd', 'acdb', 'adbc', 'adcb',
    'bacd', 'badc', 'bcad', 'bcda', 'bdac', 'bdca',
    'cabd', 'cadb', 'cbad', 'cbda', 'cdab', 'cdba',
    'dabc', 'dacb', 'dbac', 'dbca', 'dcab', 'dcba'
]

# Entropy values per dataset
entropy_data = {
    "CIFAR10-C": [0.62, 0.75, 0.88, 1.36, 0.96, 0.77, 0.89, 0.87, 1.50, 1.30, 1.53, 1.38,
                  1.36, 1.56, 1.36, 1.44, 1.44, 1.51, 1.32, 1.32, 1.35, 0.75, 1.38, 0.93],
    "CIFAR100-C": [1.20, 1.42, 1.68, 2.35, 2.01, 1.35, 1.52, 1.45, 2.53, 2.59, 2.63, 2.89,
                   2.58, 2.56, 2.58, 2.69, 2.66, 2.84, 2.67, 2.67, 2.49, 1.28, 2.81, 1.74],
    "ImageNet-C": [4.06, 4.48, 5.44, 6.48, 5.38, 4.52, 4.55, 4.63, 6.49, 6.67, 6.64, 6.56,
                   5.48, 5.55, 5.48, 5.46, 6.61, 6.64, 4.69, 4.61, 4.49, 4.53, 4.68, 5.21],
    "ImageNet-S": [3.81, 4.48, 5.53, 6.29, 5.54, 4.55, 4.75, 4.72, 6.67, 6.61, 6.64, 6.57,
                   5.25, 5.32, 5.25, 6.21, 6.12, 6.24, 5.11, 5.31, 5.29, 3.99, 5.34, 5.43],
    "ImageNet-V2": [2.24, 2.43, 3.56, 5.19, 3.67, 2.57, 2.87, 2.81, 6.11, 6.07, 6.10, 6.06,
                    3.55, 3.73, 3.55, 5.32, 5.16, 6.33, 3.50, 3.63, 3.64, 2.21, 3.71, 3.06],
    "ImageNet-R": [3.78, 3.33, 5.41, 6.29, 4.69, 3.41, 4.31, 4.10, 6.25, 6.20, 6.33, 6.24,
                   3.53, 3.81, 3.53, 5.38, 5.19, 6.53, 4.13, 4.01, 4.01, 3.12, 4.13, 5.30],
    "PACS": [0.75, 0.77, 0.86, 0.95, 0.83, 0.71, 0.81, 0.79, 1.12, 1.09, 1.13, 1.06,
             0.85, 0.91, 0.85, 1.01, 0.94, 1.09, 0.87, 0.96, 0.90, 0.68, 1.04, 0.85]
}

# Create DataFrame
df_all = pd.DataFrame(entropy_data, index=traverses).T

# Plot settings
num_datasets = len(df_all)
compact_height_per_row = 0.6
fig, axes = plt.subplots(num_datasets, 1, figsize=(18, compact_height_per_row * num_datasets + 2), sharex=True)

mappable = None

# Plot heatmaps
cols_to_highlight = [0, 1, 5, 6, 7, 21]  # columns to highlight

for ax, (dataset_name, row) in zip(axes, df_all.iterrows()):
    heatmap = sns.heatmap(
        pd.DataFrame([row.values], columns=traverses, index=[dataset_name]),
        annot=True, fmt=".2f", cmap="YlOrBr",
        cbar=False,
        linewidths=0.5, linecolor='gray',
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax
    )
    ax.set_yticklabels([dataset_name], rotation=0, fontsize=16)
    ax.tick_params(left=False, bottom=False)

    # === Add red-bordered, cross-hatched rectangles ===
    for col_index in cols_to_highlight:
        rect = patches.Rectangle(
            (col_index, 0), 1, 1,  # (x, y), width, height
            linewidth=2,
            edgecolor='green',
            facecolor='none',
            hatch='//'  # Cross hatching pattern
        )
        ax.add_patch(rect)

    # Get the mappable for the colorbar
    mappable = heatmap.get_children()[0]

# Set x-axis ticks and label
axes[-1].set_xticklabels(traverses, rotation=45, ha='right', fontsize=16)
axes[-1].set_xlabel("Permutation of Traverses", fontsize=18)

# Shared title
# plt.suptitle("Mean Entropy Heatmaps Across Datasets", fontsize=20)

# Adjust layout and colorbar
plt.tight_layout(rect=[0, 0.05, 0.98, 0.95])
# cbar_ax = fig.add_axes([0.98, 0.15, 0.015, 0.7])
cbar_ax = fig.add_axes([0.98, 0.22, 0.015, 0.71])
cbar = fig.colorbar(mappable, cax=cbar_ax)

# Remove ticks and numbers from the colorbar
cbar.ax.tick_params(
    left=False, right=False, labelleft=False, labelright=False
)
cbar.set_label("Entropy", fontsize=16)

# Save figure
plt.savefig("images/heat_combined.png", dpi=300, bbox_inches='tight')
plt.savefig("pdfs/heat_combined.pdf", dpi=300, bbox_inches='tight')
plt.close()
