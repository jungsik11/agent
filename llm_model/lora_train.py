# Copyright © 2024 Apple Inc.

import argparse, math, os, re, types, yaml 
from pathlib import Path
import numpy as np

import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, evaluate, train
from mlx_lm.tuner.utils import (
    build_schedule,
    linear_to_lora_layers,
    load_adapters,
    print_trainable_parameters,
)
from mlx_lm.utils import load, save_config

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)



def train_model(
    args,
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    train_set,
    valid_set,
    training_callback: TrainingCallback = None,
):
    model.freeze()
    if args.fine_tune_type == "full":
        for l in model.layers[-min(args.num_layers, 0) :]:
            l.unfreeze()
    elif args.fine_tune_type in ["lora", "dora"]:
        # Convert linear layers to lora/dora layers and unfreeze in the process
        linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=(args.fine_tune_type == "dora"),
        )
    else:
        raise ValueError(f"Received unknown fine-tune-type {args.fine_tune_type}")

    # Resume from weights if provided
    if args.resume_adapter_file is not None:
        print(f"Loading fine-tuned weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_file = adapter_path / "adapters.safetensors"
    save_config(vars(args), adapter_path / "adapter_config.json")

    # init training args
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=args.val_batches,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.grad_checkpoint,
    )

    model.train()
    opt = optim.Adam(
        learning_rate=(
            build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
        )
    )
    # Train model
    train(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=valid_set,
        training_callback=training_callback,
    )


def evaluate_model(args, model: nn.Module, tokenizer: TokenizerWrapper, test_set):
    model.eval()

    test_loss = evaluate(
        model=model,
        dataset=test_set,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_batches=args.test_batches,
        max_seq_length=args.max_seq_length,
    )

    test_ppl = math.exp(test_loss)

    print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


def run(args, training_callback: TrainingCallback = None):
    np.random.seed(args.seed)

    print("Loading pretrained model")
    model, tokenizer = load(args.model)

    print("Loading datasets")
    train_set, valid_set, test_set = load_dataset(args, tokenizer)

    if args.test and not args.train:
        # Allow testing without LoRA layers by providing empty path
        if args.adapter_path != "":
            load_adapters(model, args.adapter_path)

    elif args.train:
        print("Training")
        train_model(args, model, tokenizer, train_set, valid_set, training_callback)
    else:
        raise ValueError("Must provide at least one of --train or --test")

    if args.test:
        print("Testing")
        evaluate_model(args, model, tokenizer, test_set)


def lora_train(config_file, model, dataset,adapter):
    args = {}
    args['train'] = True
    args['completion_feature'] = True
    

    if config_file:
        print("Loading configuration file", config_file)
        
        with open(config_file, "r") as file:
            config_file = yaml.load(file, yaml_loader)

        # Prefer parameters from command-line arguments
        for k, v in config_file.items():
            if args.get(k, None) is None:
                args[k] = v
    if model != None: args['model'] = model
    if dataset != None: args['data'] = dataset
    if adapter != None: args['adapter_path'] = adapter
    print( args)
    #run(types.SimpleNamespace(**args))
