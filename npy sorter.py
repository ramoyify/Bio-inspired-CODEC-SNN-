import os
import shutil

# Define the paths
results_folder = "results"
npy_results_folder = "npy_results"

# Create the npy_results folder if it doesn't exist
if not os.path.exists(npy_results_folder):
    os.makedirs(npy_results_folder)

# Iterate over files in the results folder
for filename in os.listdir(results_folder):
    # Check if the file is an npy file
    if filename.endswith(".npy"):
        # Create the full file paths
        src = os.path.join(results_folder, filename)
        dst = os.path.join(npy_results_folder, filename)
        # Move the file to npy_results folder
        shutil.move(src, dst)

print("Npy files moved successfully.")
