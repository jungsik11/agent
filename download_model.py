
from huggingface_hub import hf_hub_download
import os

# Define repository and filename
repo_id = "beomi/gemma-ko-2b-gguf"
filename = "gemma-ko-2b-f16.gguf"  # Using the float16 version

# Define the local directory to save the model
local_dir = "/Users/irf/Desktop/SynologyDrive/git/agent/train_example/data"

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading {filename} from {repo_id}...")

# Download the model file
hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    local_dir_use_symlinks=False # Download the actual file, not a symlink
)

print(f"Model downloaded successfully to {os.path.join(local_dir, filename)}")
