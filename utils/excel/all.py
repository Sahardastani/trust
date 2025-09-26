import os
import re
import pandas as pd

# Define input and output paths
input_folder = "utils/results/"  # Change to your actual folder path
output_file = "utils/excel/output.xlsx"

# Define regex pattern for extracting overall accuracy and corruption type
accuracy_pattern = r"\[.*?\] Overall Accuracy: ([\d.]+)%\s+\|\s+Corruption: (\w+)"

# Define regex patterns for extracting row name components
dataset_pattern = r"dataset: (\w+)"
episodic_pattern = r"episodic: (\w+)"
model_merging_pattern = r"model_merging: (\w+)"
weight_list_pattern = r"weight_list: \[(.*?)\]"

# Predefined order of corruption types
corruption_order = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", 
    "motion_blur", "zoom_blur", "frost", "snow", "fog", "brightness", "contrast", 
    "elastic_transform", "pixelate", "jpeg_compression"
]

# Initialize a dictionary to store results
data_dict = {}

# Process each .txt file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):  # Process only .txt files
        file_path = os.path.join(input_folder, filename)
        
        # Read file content
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.readlines()
        
        # Extract metadata for row name
        dataset, episodic, model_merging, weight_count = None, None, None, None
        for line in content:
            if dataset is None and (match := re.search(dataset_pattern, line)):
                dataset = match.group(1)
            if episodic is None and (match := re.search(episodic_pattern, line)):
                episodic = match.group(1)
            if model_merging is None and (match := re.search(model_merging_pattern, line)):
                model_merging = match.group(1)
            if weight_count is None and (match := re.search(weight_list_pattern, line)):
                weight_count = len(match.group(1).split(","))  # Count elements in list
        
        # Construct row name
        if dataset and episodic and model_merging and weight_count is not None:
            row_name = f"{dataset}_{episodic}_{model_merging}_{weight_count}"
        else:
            continue  # Skip if any metadata is missing
        
        # Extract accuracy data
        accuracies = {}
        for line in content:
            match = re.search(accuracy_pattern, line)
            if match:
                accuracy = float(match.group(1))  # Convert accuracy to float
                corruption = match.group(2)  # Extract corruption type
                accuracies[corruption] = accuracy
        
        # Store extracted data
        data_dict[row_name] = accuracies

# Create DataFrame
df = pd.DataFrame.from_dict(data_dict, orient='index', columns=corruption_order)

# Save to Excel
df.to_excel(output_file, index=True, index_label="Experiment")

print(f"Data successfully saved to {output_file}")
