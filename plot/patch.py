# # Create 3x3 grid of patches with numbers
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Rectangle

# # Load the image
# image_path = "/home/as89480/vssm_TTA/plot/images/snake.png"
# img = Image.open(image_path)

# # Convert to numpy array
# img_array = np.array(img)

# # Calculate patch size
# h, w, _ = img_array.shape
# patch_h, patch_w = h // 3, w // 3

# fig, axes = plt.subplots(3, 3, figsize=(6, 6))
# patch_number = 1
# for i in range(3):
#     for j in range(3):
#         patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
#         axes[i, j].imshow(patch)
#         axes[i, j].axis('off')
#         axes[i, j].text(patch_w // 2, patch_h // 2, str(patch_number),
#                         color='white', fontsize=20, ha='center', va='center',
#                         bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
#         patch_number += 1

# plt.tight_layout()
# plt.savefig('0.png')

# ######################################################

# reordered_grid_positions = [
#     [(0, 0), (1, 0), (2, 0)],  # row 1
#     [(0, 1), (1, 1), (2, 1)],  # row 2
#     [(0, 2), (1, 2), (2, 2)]   # row 3
# ]

# # Create the reordered grid
# fig, axes = plt.subplots(3, 3, figsize=(6, 6))
# for row_idx, row in enumerate(reordered_grid_positions):
#     for col_idx, (i, j) in enumerate(row):
#         patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
#         patch_number = i * 3 + j + 1
#         axes[row_idx, col_idx].imshow(patch)
#         axes[row_idx, col_idx].axis('off')
#         axes[row_idx, col_idx].text(patch_w // 2, patch_h // 2, str(patch_number),
#                                     color='white', fontsize=20, ha='center', va='center',
#                                     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

# plt.tight_layout()
# plt.savefig('1.png')

# ######################################################

# reordered_grid_positions_reverse = [
#     [(2, 2), (2, 1), (2, 0)],  # row 1
#     [(1, 2), (1, 1), (1, 0)],  # row 2
#     [(0, 2), (0, 1), (0, 0)]   # row 3
# ]

# # Create the reversed grid
# fig, axes = plt.subplots(3, 3, figsize=(6, 6))
# for row_idx, row in enumerate(reordered_grid_positions_reverse):
#     for col_idx, (i, j) in enumerate(row):
#         patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
#         patch_number = i * 3 + j + 1
#         axes[row_idx, col_idx].imshow(patch)
#         axes[row_idx, col_idx].axis('off')
#         axes[row_idx, col_idx].text(patch_w // 2, patch_h // 2, str(patch_number),
#                                     color='white', fontsize=20, ha='center', va='center',
#                                     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

# plt.tight_layout()
# plt.savefig('2.png')

# ######################################################

# reordered_grid_positions_columnwise_reverse = [
#     [(2, 2), (1, 2), (0, 2)],  # row 1
#     [(2, 1), (1, 1), (0, 1)],  # row 2
#     [(2, 0), (1, 0), (0, 0)]   # row 3
# ]

# # Create the new grid
# fig, axes = plt.subplots(3, 3, figsize=(6, 6))
# for row_idx, row in enumerate(reordered_grid_positions_columnwise_reverse):
#     for col_idx, (i, j) in enumerate(row):
#         patch = img_array[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
#         patch_number = i * 3 + j + 1
#         axes[row_idx, col_idx].imshow(patch)
#         axes[row_idx, col_idx].axis('off')
#         axes[row_idx, col_idx].text(patch_w // 2, patch_h // 2, str(patch_number),
#                                     color='white', fontsize=20, ha='center', va='center',
#                                     bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

# plt.tight_layout()
# plt.savefig('3.png')

# ######################################################

# from PIL import Image
# import matplotlib.pyplot as plt

# # List of saved image paths
# image_paths = ['0.png', '1.png', '2.png', '3.png']

# # Load images
# images = [Image.open(path) for path in image_paths]

# # Create a horizontal plot
# fig, axes = plt.subplots(1, 4, figsize=(15, 4.1))  # Adjust figsize as needed

# for ax, img, idx in zip(axes, images, range(4)):
#     ax.imshow(img)
#     ax.axis('off')
#     ax.set_title(f"Traverse {idx}")

# plt.tight_layout()
# plt.savefig('images/traverses.png')

# ######################################################

# # List of saved image paths
# image_paths = ['0.png', '1.png', '2.png', '3.png']

# # Corresponding letters
# letters = ['a', 'b', 'c', 'd']

# # Load images
# images = [Image.open(path) for path in image_paths]

# # Create a horizontal plot
# fig, axes = plt.subplots(1, 4, figsize=(15, 4.1))

# for ax, img, letter in zip(axes, images, letters):
#     ax.imshow(img)
#     ax.axis('off')
#     ax.set_title(f"Traverse {letter}", fontsize=15)

#     # Add a black border around each image
#     rect = Rectangle((0, 0), 1, 1, transform=ax.transAxes,
#                      linewidth=3, edgecolor='black', facecolor='none')
#     ax.add_patch(rect)

# plt.tight_layout()
# plt.savefig('images/traverses_boarder.png', dpi=300)
