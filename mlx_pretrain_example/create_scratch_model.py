import json
import sys
import types
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.llama import Model
from mlx_lm.utils import load_config
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def main():
    """
    This script creates a new Llama-type model with random weights and a
    trained tokenizer, preparing it for pre-training from scratch.
    """
    model_path = Path("scratch_model")
    if not model_path.exists() or not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"The model config directory '{model_path}' with a 'config.json' file does not exist."
        )

    print("1. Loading model configuration...")
    config_dict = load_config(model_path)
    config = types.SimpleNamespace(**config_dict)

    print("2. Initializing model with random weights...")
    model = Model(config)

    print("3. Inspecting model parameters before saving...")
    params = model.parameters()
    for top_key, sub_dict in params.items():
        print(f"- Top-level key: '{top_key}', Type: {type(sub_dict)}")
        if isinstance(sub_dict, dict):
            for sub_key, value in sub_dict.items():
                print(f"  - Sub-key: '{sub_key}', Value Type: {type(value)}")

    print(f"\nAttempting to save random weights to '{model_path / 'model.safetensors'}'...")
    # Recursively flatten the parameters dictionary, handling nested lists and dicts
    def flatten_params(params, prefix=''):
        flat_dict = {}
        if isinstance(params, dict):
            for k, v in params.items():
                flat_dict.update(flatten_params(v, prefix + k + '.'))
        elif isinstance(params, list):
            for i, v in enumerate(params):
                flat_dict.update(flatten_params(v, prefix + str(i) + '.'))
        else:
            # Remove the trailing dot from the key
            flat_dict[prefix[:-1]] = params
        return flat_dict

    flat_params = flatten_params(model.parameters())
    mx.save_safetensors(str(model_path / "model.safetensors"), flat_params)

    print("4. Training and saving tokenizer...")
    # Use the 'tokenizers' library to train and save the tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=config.vocab_size)

    files = [str(f) for f in Path("data/").glob("*.jsonl")]
    tokenizer.train(files, trainer)

    tokenizer_save_path = model_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_save_path))
    print(f"   > Tokenizer saved to '{tokenizer_save_path}'")

    print("5. Creating 'tokenizer_config.json'...")
    tokenizer_config = {"model_type": "llama", "tokenizer_class": "LlamaTokenizer"}
    with open(model_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    print("\nScratch model created successfully.")


if __name__ == "__main__":
    main()
