import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import os

from .qwen3_moe import Model, ModelArgs, QWEN3_MOE_1B_ARGS

def count_parameters(model):
    """Counts the total number of parameters in a model."""
    return sum(x[1].size for x in tree_flatten(model.parameters()))

def inspect_model_info(model_path, is_quantized_path=False):
    """
    Loads a model and prints its information, including parameter count.
    """
    print(f"\n--- Inspecting Model: {model_path} ---")
    
    # Load model arguments
    model_args = ModelArgs(**QWEN3_MOE_1B_ARGS)
    model = Model(model_args)
    
    # If loading a quantized model, quantize the model object first
    if is_quantized_path:
        print("Quantizing model object to prepare for loading quantized weights...")
        nn.quantize(model, 64, 4) # Using supported group_size=64 and bits=4
        
    # Load weights
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_weights(model_path)
    else:
        print(f"Model weights not found at {model_path}. Skipping parameter count.")
        return

    # Print parameter count
    num_params = count_parameters(model)
    print(f"Total Parameters: {num_params:,}")

if __name__ == "__main__":
    original_model_path = "train_example/qwen3_moe_1B_korean/model.safetensors"
    quantized_model_path = "train_example/qwen3_moe_1B_korean/model_quantized.safetensors"

    inspect_model_info(original_model_path, is_quantized_path=False)
    inspect_model_info(quantized_model_path, is_quantized_path=True)
