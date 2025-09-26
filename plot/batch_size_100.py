import matplotlib.pyplot as plt

batch_sizes = [16, 32, 64, 128]

# CIFAR100
tent_values = [40.77, 41.56, 41.79, 41.8]
ours_values = [47.34, 52.53, 54.47, 54.3]

dataset = "CIFAR100"

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(batch_sizes, ours_values, marker='o', label='Ours')
plt.plot(batch_sizes, tent_values, marker='o', label='Tent')
plt.xlabel('Batch Size', fontsize=18)
plt.ylabel('Accuracy (%)', fontsize=18)
plt.title(f'Accuracy vs. Batch Size on {dataset}-C', fontsize=20)
plt.xticks(batch_sizes, labels=['16', '32', '64', '128'], fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, linestyle=':')
plt.legend(fontsize=16)
plt.tight_layout()

plt.savefig(f"images/batch_size_{dataset}.png")
plt.savefig(f"pdfs/batch_size_{dataset}.pdf")
