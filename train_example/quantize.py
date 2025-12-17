import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import os
from .qwen3_moe import Model, ModelArgs, QWEN3_MOE_1B_ARGS

def quantize_model(model_path, quantized_model_path, q_bits=4, q_group_size=64):
    """
    Loads a model, quantizes it, and saves the quantized model.
    """
    print(f"Loading model from {model_path}...")
    model = Model(ModelArgs(**QWEN3_MOE_1B_ARGS))
    model.load_weights(model_path)

    print(f"Quantizing model with bits={q_bits} and group_size={q_group_size}...")
    nn.quantize(model, q_group_size, q_bits)

    print(f"Saving quantized model to {quantized_model_path}...")
    # There is no direct way to save a quantized model's state dict.
    # We need to get the quantized parameters and save them.
    # The `tree_flatten` utility can be used to get the parameters.
    quantized_weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(quantized_model_path, quantized_weights)
    print("Quantization complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantize a Qwen3 MoE model.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="train_example/qwen3_moe_1B_korean/model.safetensors",
        help="Path to the trained model.",
    )
    parser.add_argument(
        "--quantized-model-path",
        type=str,
        default="train_example/qwen3_moe_1B_korean/model_quantized.safetensors",
        help="Path to save the quantized model.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Bits for quantization.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Group size for quantization.",
    )
    args = parser.parse_args()

    quantize_model(
        args.model_path,
        args.quantized_model_path,
        args.bits,
        args.group_size,
    )
