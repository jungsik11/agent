from datasets import load_dataset
import os

# Define the dataset name on Hugging Face
dataset_name = "kmgme/KorQuADv2_1"
output_dir = "data/korquad_v2_1"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Downloading {dataset_name}...")

# Load the dataset
# This will download the dataset to your Hugging Face cache directory
dataset = load_dataset(dataset_name)

# Save the dataset to local JSON files
print(f"Saving dataset to {output_dir}...")
for split in dataset.keys():
    file_path = os.path.join(output_dir, f"{split}.json")
    dataset[split].to_json(file_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved {split} split to {file_path}")

print("Download and save complete.")
