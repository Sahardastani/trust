import os
import pandas as pd

def extract_metrics_from_txt(file_path):
    """Extract exactly 15 overall accuracy and 15 mean entropy values from a text file."""
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    overall_accuracies = []
    mean_entropies = []
    
    for line in lines:
        if "Overall Accuracy:" in line:
            overall_accuracy = float(line.split("Overall Accuracy:")[1].split("%")[0].strip())
            overall_accuracies.append(overall_accuracy)
        if "Mean Entropy:" in line:
            mean_entropy = float(line.split("Mean Entropy:")[1].split("|")[0].strip())
            mean_entropies.append(mean_entropy)
    
    # Ensure exactly 15 values are captured
    overall_accuracies = overall_accuracies[:15]
    mean_entropies = mean_entropies[:15]
    
    # Pad with NaN if fewer than 15 values are found
    while len(overall_accuracies) < 15:
        overall_accuracies.append(float("nan"))
    while len(mean_entropies) < 15:
        mean_entropies.append(float("nan"))
    
    # Interleave accuracy and entropy values
    interleaved_values = [val for pair in zip(overall_accuracies, mean_entropies) for val in pair]
    return interleaved_values

def process_all_txt_files(folder_path, output_xlsx):
    """Process all txt files in a folder and save extracted metrics into an Excel file."""
    all_data = []
    file_names = []
    
    # Get sorted list of txt files
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
    
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        extracted_values = extract_metrics_from_txt(file_path)
        if extracted_values:
            file_names.append(file_name)
            all_data.append(extracted_values)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, index=file_names)
    
    # Save to Excel
    df.to_excel(output_xlsx, index=True, header=False)
    print(f"Extracted data saved to {output_xlsx}")

# Folder containing txt files
folder_path = "/home/as89480/vssm_TTA/utils/results/"
# Output Excel file path
output_xlsx = "/home/as89480/vssm_TTA/utils/excel/mean.xlsx"

# Run the extraction process
process_all_txt_files(folder_path, output_xlsx)