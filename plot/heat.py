import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# traverses = [
#     'abcd', 'abdc', 'acbd', 'acdb', 'adbc', 'adcb',
#     'bacd', 'badc', 'bcad', 'bcda', 'bdac', 'bdca',
#     'cabd', 'cadb', 'cbad', 'cbda', 'cdab', 'cdba',
#     'dabc', 'dacb', 'dbac', 'dbca', 'dcab', 'dcba'
# ]

traverses = [
    r'$\pi_1 = abcd$', r'$\pi_2 = abdc$', r'$\pi_3 = acbd$', r'$\pi_4 = acdb$', r'$\pi_5 = adbc$', r'$\pi_6 = adcb$',
    r'$\pi_7 = bacd$', r'$\pi_8 = badc$', r'$\pi_9 = bcad$', r'$\pi_{10} = bcda$', r'$\pi_{11} = bdac$', r'$\pi_{12} = bdca$',
    r'$\pi_{13} = cabd$', r'$\pi_{14} = cadb$', r'$\pi_{15} = cbad$', r'$\pi_{16} = cbda$', r'$\pi_{17} = cdab$', r'$\pi_{18} = cdba$',
    r'$\pi_{19} = dabc$', r'$\pi_{20} = dacb$', r'$\pi_{21} = dbac$', r'$\pi_{22} = dbca$', r'$\pi_{23} = dcab$', r'$\pi_{24} = dcba$'
]


imagenet_entropy = [
    4.06, 4.53, 5.44, 6.36, 6.48, 5.0,
    4.47, 4.72, 6.45, 6.69, 6.67, 6.54,
    6.4, 6.56, 5.48, 6.46, 5.93, 6.61,
    6.68, 6.47, 6.52, 4.53, 6.6, 5.21
]

# Create a DataFrame (1 row: ImageNet-C)
df = pd.DataFrame([imagenet_entropy], index=["ImageNet-C"], columns=traverses)

# Create a 2D array of traverses to annotate
annot_array = np.array([traverses])  # Shape must match df shape (1, 24)

# Create a PDF and PNG
with PdfPages('pdfs/heatmap.pdf') as pdf:
    # Plot heatmap
    plt.figure(figsize=(14, 2.5))
    ax = sns.heatmap(df, annot=annot_array, fmt="s", cmap="YlOrBr",
                     cbar_kws={'label': 'Mean Entropy'},
                     linewidths=1, linecolor='gray', cbar=True,
                     annot_kws={"size": 20, "weight": "bold", "rotation": 90})  # <<<< ROTATE 90 degrees

    # Adjust the color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Mean Entropy', fontsize=13)

    # Set titles and labels
    plt.title("Mean Entropy Heatmap – ImageNet-C", fontsize=15)
    plt.ylabel("", fontsize=13)
    plt.xticks([], [])  # Hides the x-axis labels
    plt.yticks(fontsize=13)

    plt.tight_layout()

    # Save the plot to the PDF
    pdf.savefig(bbox_inches='tight')

    # Save the same plot as a PNG image
    plt.savefig("images/heatmap.png", dpi=300, bbox_inches='tight')

    # Add metadata to PDF
    d = pdf.infodict()
    d['Title'] = 'Centered Heatmap PDF'
    pdf.attach_note("Plot centered on the page", positionRect=[-1, -1, -1, -1])

    # Close the figure to free memory
    plt.close()
