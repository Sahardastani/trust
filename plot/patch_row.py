from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

# Load the image
image_path = "/home/as89480/vssm_TTA/plot/images/snake.png"
img = Image.open(image_path)

# Convert to numpy array
img_array = np.array(img)

# # Calculate patch size
# h, w, _ = img_array.shape
# patch_h, patch_w = h // 3, w // 3

# # ========== 1. Regular 3x3 grid in row ==========
# fig, axes = plt.subplots(1, 9, figsize=(18, 2))
# patch_number = 1
# for i in range(3):
#     for j in range(3):
#         patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
#         ax = axes[(i * 3) + j]
#         ax.imshow(patch)
#         ax.axis('off')
#         ax.text(patch_w // 2, patch_h // 2, str(patch_number),
#                 color='white', fontsize=12, ha='center', va='center',
#                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
#         patch_number += 1
# plt.tight_layout()
# plt.savefig('images/0_row.png')


# # ========== 2. Reordered column-wise (left to right) ==========
# reordered_grid_positions = [
#     [(0, 0), (1, 0), (2, 0)],
#     [(0, 1), (1, 1), (2, 1)],
#     [(0, 2), (1, 2), (2, 2)]
# ]

# fig, axes = plt.subplots(1, 9, figsize=(18, 2))
# idx = 0
# for row in reordered_grid_positions:
#     for i, j in row:
#         patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
#         patch_number = i * 3 + j + 1
#         ax = axes[idx]
#         ax.imshow(patch)
#         ax.axis('off')
#         ax.text(patch_w // 2, patch_h // 2, str(patch_number),
#                 color='white', fontsize=12, ha='center', va='center',
#                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
#         idx += 1
# plt.tight_layout()
# plt.savefig('images/1_row.png')


# # ========== 3. Reversed grid ==========
# reordered_grid_positions_reverse = [
#     [(2, 2), (2, 1), (2, 0)],
#     [(1, 2), (1, 1), (1, 0)],
#     [(0, 2), (0, 1), (0, 0)]
# ]

# fig, axes = plt.subplots(1, 9, figsize=(18, 2))
# idx = 0
# for row in reordered_grid_positions_reverse:
#     for i, j in row:
#         patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
#         patch_number = i * 3 + j + 1
#         ax = axes[idx]
#         ax.imshow(patch)
#         ax.axis('off')
#         ax.text(patch_w // 2, patch_h // 2, str(patch_number),
#                 color='white', fontsize=12, ha='center', va='center',
#                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
#         idx += 1
# plt.tight_layout()
# plt.savefig('images/2_row.png')


# # ========== 4. Columnwise reversed ==========
# reordered_grid_positions_columnwise_reverse = [
#     [(2, 2), (1, 2), (0, 2)],
#     [(2, 1), (1, 1), (0, 1)],
#     [(2, 0), (1, 0), (0, 0)]
# ]

# fig, axes = plt.subplots(1, 9, figsize=(18, 2))
# idx = 0
# for row in reordered_grid_positions_columnwise_reverse:
#     for i, j in row:
#         patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
#         patch_number = i * 3 + j + 1
#         ax = axes[idx]
#         ax.imshow(patch)
#         ax.axis('off')
#         ax.text(patch_w // 2, patch_h // 2, str(patch_number),
#                 color='white', fontsize=12, ha='center', va='center',
#                 bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
#         idx += 1
# plt.tight_layout()
# plt.savefig('images/3_row.png')


################################################

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load the image
image_path = "/home/as89480/vssm_TTA/plot/images/snake.png"
img = Image.open(image_path)
img_array = np.array(img)

# Patch size
h, w, _ = img_array.shape
patch_h, patch_w = h // 3, w // 3

# Define all grid configurations
grids = [
    [(i, j) for i in range(3) for j in range(3)],  # traverse a: original
    [(i, j) for j in range(3) for i in range(3)],  # traverse b: column-major
    [(i, j) for i, j in [(2,2), (2,1), (2,0), (1,2), (1,1), (1,0), (0,2), (0,1), (0,0)]],  # traverse c: reversed
    [(i, j) for i, j in [(2,2), (1,2), (0,2), (2,1), (1,1), (0,1), (2,0), (1,0), (0,0)]]   # traverse d: column-wise reverse
]

traverse_labels = ["Traverse a", "Traverse b", "Traverse c", "Traverse d"]

fig, axes = plt.subplots(4, 9, figsize=(15, 8))
# plt.subplots_adjust(left=0.08)

for row_idx, (grid, label) in enumerate(zip(grids, traverse_labels)):
    for col_idx, (i, j) in enumerate(grid):
        patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
        patch_number = i * 3 + j + 1
        ax = axes[row_idx, col_idx]
        ax.imshow(patch)
        ax.axis('off')

        # Add label number
        ax.text(patch_w // 2, patch_h // 2, str(patch_number),
                color='white', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

        # Add rectangle border around the patch
        # rect = patches.Rectangle((0, 0), 1, 1,
        #                          linewidth=2, edgecolor='black', facecolor='none',
        #                          transform=ax.transAxes, clip_on=False)
        # ax.add_patch(rect)

        # === Add a border around the entire row ===
        # Get positions of the first and last axes in this row
        first_ax = axes[row_idx, 0]
        last_ax = axes[row_idx, -1]
        row_y0 = first_ax.get_position().y0
        row_y1 = first_ax.get_position().y1
        row_x0 = first_ax.get_position().x0
        row_x1 = last_ax.get_position().x1

        # Draw rectangle around the full row
        row_rect = patches.Rectangle((row_x0, row_y0),
                                    row_x1 - row_x0, row_y1 - row_y0,
                                    linewidth=2, edgecolor='black', facecolor='none',
                                    transform=fig.transFigure, clip_on=False)
        fig.patches.append(row_rect)


    # Add label next to each row
        # Add label to the right of each row
    row_ax = axes[row_idx, -1]
    font_props = fm.FontProperties(family='sans-serif', weight='medium', size=20)

    # In your fig.text call:
    fig.text(row_ax.get_position().x1 + 0.005,
            (row_ax.get_position().y0 + row_ax.get_position().y1) / 2,
            label, va='center', ha='left', fontproperties=font_props)

plt.savefig('images/4_rows_together.png', bbox_inches='tight', dpi=300)


################################################


image_path = "/home/as89480/vssm_TTA/plot/images/snake.png"
img = Image.open(image_path)

# Convert to numpy array
img_array = np.array(img)

# Calculate patch size
h, w, _ = img_array.shape
patch_h, patch_w = h // 3, w // 3

fig, axes = plt.subplots(3, 3, figsize=(6, 6))
patch_number = 1
for i in range(3):
    for j in range(3):
        patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
        axes[i, j].imshow(patch)
        axes[i, j].axis('off')
        axes[i, j].text(patch_w // 2, patch_h // 2, str(patch_number),
                        color='white', fontsize=20, ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
        patch_number += 1

plt.tight_layout()
plt.savefig('images/0.png')

##################################################3


from PIL import Image
import numpy as np

# Paths to your images
traversal_img_path = "images/4_rows_together.png"  # previously generated
full_img_path = "images/0.png"  # full snake image

# Load images
full_img = Image.open(full_img_path)
traversal_img = Image.open(traversal_img_path)

# Resize full image to ~85% of traversal height
target_height = int(traversal_img.height * 1)
aspect_ratio = full_img.width / full_img.height
new_width = int(target_height * aspect_ratio)
full_img_resized = full_img.resize((new_width, target_height), Image.Resampling.LANCZOS)

# Pad full image vertically to match traversal height
top_pad = (traversal_img.height - full_img_resized.height) // 2
padded_full_img = Image.new("RGB", (new_width, traversal_img.height), (255, 255, 255))
padded_full_img.paste(full_img_resized, (0, top_pad))

# === Reduce the gap between images ===
gap = 0  # pixels of horizontal spacing (adjust as needed)
combined_width = new_width + gap + traversal_img.width

# Create combined image
combined_img = Image.new("RGB", (combined_width, traversal_img.height), (255, 255, 255))
combined_img.paste(padded_full_img, (0, 0))
combined_img.paste(traversal_img, (new_width + gap, 0))

# Save and show
combined_img.save("images/combined_snakes.png")
combined_img.show()
