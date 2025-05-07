from pathlib import Path
from typing import Callable
from mlx_lm import load
import argparse, glob,shutil, yaml 
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.tuner.utils import dequantize, load_adapters
from mlx_lm.utils import (
    dequantize_model,
    fetch_from_hub,
    get_model_path,
    quantize_model,
    save_config,
    save_weights,
)


def mixed_quant_predicate_builder(
    recipe: str, model: nn.Module
) -> Callable[[str, nn.Module, dict], Union[bool, dict]]:

    if recipe == "mixed_2_6":
        low_bits = 2
    elif recipe == "mixed_3_6":
        low_bits = 3
    elif recipe == "mixed_4_6":
        low_bits = 4
    else:
        raise ValueError("Invalid quant recipe {recipe}")
    high_bits = 6
    group_size = 64

    down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
    if len(down_keys) == 0:
        raise ValueError("Model does not have expected keys for mixed quant.")

    # Look for the layer index location in the path:
    for layer_location, k in enumerate(down_keys[0].split(".")):
        if k.isdigit():
            break
    num_layers = len(model.layers)

    def mixed_quant_predicate(
        path: str,
        module: nn.Module,
        config: dict,
    ) -> Union[bool, dict]:
        """Implements mixed quantization predicates with similar choices to, for example, llama.cpp's Q4_K_M.
        Ref: https://github.com/ggerganov/llama.cpp/blob/917786f43d0f29b7c77a0c56767c0fa4df68b1c5/src/llama.cpp#L5265
        By Alex Barron: https://gist.github.com/barronalex/84addb8078be21969f1690c1454855f3
        """

        if not hasattr(module, "to_quantized"):
            return False

        index = (
            int(path.split(".")[layer_location])
            if len(path.split(".")) > layer_location
            else 0
        )
        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )
        if "v_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits}
        if "down_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits}
        if "lm_head" in path:
            return {"group_size": group_size, "bits": high_bits}

        return {"group_size": group_size, "bits": low_bits}

    return mixed_quant_predicate



def configure_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument("--model", type=str, required =True, help="Path to the Hugging Face model.")
    parser.add_argument("--adapter-path", type=str, default="mlx_model", help="Adapter path."    )
    parser.add_argument( "--quantize", type=bool, default=False)
    parser.add_argument( "--dequantize", type=bool, default=False)

    return parser



def main():
    parser = configure_parser()
    #args = parser.parse_args(args=[]) # for jupyter notebook
    args = parser.parse_args()

    hf_path = args.model

    current_path = Path.cwd()

    with open(str(Path(current_path /'template' /'lora_config.yaml')), 'r') as file:
        configs = yaml.load(file, yaml.SafeLoader)
    configs['model']= hf_path.split("/")[-1]

    model_list_path = str(Path(current_path /'models'/'config' /'config.yaml'))
    with open(model_list_path, 'r') as file:
        model_list = yaml.load(file, yaml.SafeLoader)
    if model_list['models'] == None:
        model_list['models'] = []
    if configs['model'] not in model_list['models']:
        model_list['models'].append(configs['model'])
    

    with open(model_list_path, 'w') as fp:
        yaml.dump(model_list, fp, default_flow_style=False, allow_unicode=True)
    fp.close()


    save_path = Path( current_path/'models' /'model_repo' /configs['model'])

    model_save_path = Path( save_path /'model')
    config_save_path = Path( save_path /'config')
    adapters_save_path = Path( save_path /'adapters')

    dtype: str = "float16"
    q_group_size: int = 64
    q_bits: int = 4
    quant_predicate: Optional[Union[Callable[[str, nn.Module, dict], Union[bool, dict]], str]]=None

    print("[INFO] Loading")
    model_path = get_model_path(hf_path)
    model, config, tokenizer = fetch_from_hub(model_path=model_path)
    
    if isinstance(quant_predicate, str):
        quant_predicate = mixed_quant_predicate_builder(quant_predicate, model)

    weights = dict(tree_flatten(model.parameters()))
    dtype = getattr(mx, dtype)
    if hasattr(model, "cast_predicate"):
        cast_predicate = model.cast_predicate()
    else:
        cast_predicate = lambda _: True
    weights = {
        k: v.astype(dtype) if cast_predicate(k) else v for k, v in weights.items()
    }

    # Check the save path is empty
    mlx_path = Path(model_save_path)

    #if mlx_path.exists():
    #    raise ValueError(
    #        f"Cannot save to the path {mlx_path} as it already exists."
    #        " Please delete the file/directory or specify a new path to save to."
    #    )

    if args.quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if args.quantize:
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(
            model, config, q_group_size, q_bits, quant_predicate=quant_predicate
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))

    del model
    save_weights(mlx_path, weights, donate_weights=True)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    save_config(config, config_path=mlx_path / "config.json")
    config_save_path.mkdir(parents=True, exist_ok=True)
    adapters_save_path.mkdir(parents=True, exist_ok=True)
    #configs['dtype'] =  dtype
    configs['q_group_size'] = q_group_size
    configs['q_bits'] = q_bits
    configs['quant_predicate'] =quant_predicate

    with open(str(config_save_path)+'/config.yaml', 'w') as fp:
        yaml.dump(configs, fp, default_flow_style=False, allow_unicode=True)
    fp.close()
    
    print("Hello from agent!")


if __name__ == "__main__":
    main()

# python ./models/save_model.py --model "mlx-community/Llama-3.2-1B-Instruct-bf16"
#python ./models/save_model.py --model "mlx-community/Josiefied-Qwen3-0.6B-abliterated-v1-bf16"
#python ./models/save_model.py --model "mlx-community/gemma-3-1b-it-qat-4bit"