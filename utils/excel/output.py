import re
import pandas as pd

file_name = 'tent_cifar10c_1743781912'
# Define the input file path
input_file = f"utils/results/{file_name}.txt"  # Change this to your actual file path
output_file = "utils/excel/output.xlsx"

# Define the regex pattern
pattern = r"\[.*?\] Overall Accuracy: ([\d.]+)%\s+\|\s+Corruption: (\w+)"

# Initialize a list to store results
data = []
corruption_types = []
accuracies = []

# Read the file and extract information
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            accuracy = float(match.group(1))  # Extracted accuracy as float
            corruption = match.group(2)  # Extracted corruption type
            corruption_types.append(corruption)
            accuracies.append(accuracy)

# Create a DataFrame in horizontal format
df = pd.DataFrame([accuracies], columns=corruption_types)

# Save to Excel
df.to_excel(output_file, index=False)

print(f"Data successfully saved to {output_file}")
