import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sentencepiece as spm
from datasets import load_dataset
from mlx.core import save_safetensors
from mlx.utils import tree_flatten
import sys
from tqdm import tqdm

# Import ModelArgs and Model from gemma3.py (do not modify gemma3.py)
from gemma3 import ModelArgs, Model


def get_args():
    """Parse command line arguments with default values."""
    default_model_dir = "/Users/irf/Desktop/SynologyDrive/git/agent/train_example/gemma3_3B_korean_realqa_reasoning"

    parser = argparse.ArgumentParser(description="Train a Gemma-3 model with MLX.")
    parser.add_argument("--model-dir", type=str, default=default_model_dir,
                        help="Directory to save/load model and tokenizer.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length for training.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.") # Further reduced LR
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--save-every", type=int, default=5, help="Save a checkpoint every N steps.")
    parser.add_argument("--fresh-start", action="store_true", default=False, # Set fresh-start to True by default
                        help="Force training from scratch, ignoring existing checkpoints.")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum number of training steps. Overrides epochs.")
    parser.add_argument("--num-train-examples", type=int, default=None, help="Number of training examples to use. If None, uses all available.")
    return parser.parse_args()


def get_tokenizer(model_dir: str):
    """Load the SentencePiece tokenizer."""
    tokenizer_path = os.path.join(model_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Please ensure it exists.")
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    return tokenizer


def load_dataset_and_preprocess(args, tokenizer, seq_len: int):
    """Load and preprocess the local Korean textbooks QA dataset."""
    print("Loading and preprocessing local dataset...")
    dataset_path = "/Users/irf/Desktop/SynologyDrive/git/agent/train_example/data/korean_textbooks_qa"
    try:
        dataset = load_from_disk(dataset_path)
        if isinstance(dataset, dict) and 'train' in dataset:
            dataset = dataset['train']
        print(f"Dataset loaded. Number of examples: {len(dataset)}")
    except Exception as e:
        print(f"Failed to load dataset from {dataset_path}: {e}")
        print("Attempting to load as a single arrow file...")
        arrow_file_path = os.path.join(dataset_path, "data-00000-of-00001.arrow")
        try:
            dataset = load_dataset("arrow", data_files={'train': arrow_file_path})['train']
            print(f"Dataset loaded from arrow file. Number of examples: {len(dataset)}")
        except Exception as arrow_e:
            print(f"Failed to load dataset from arrow file: {arrow_e}")
            return None

    def tokenize_fn(example):
        question = (example.get('question', '') or '').strip()
        answer = (example.get('answer', '') or '').strip()
        
        if not question or not answer:
            return {"input_ids": [], "target_ids": []}
        
        full_text = f"질문:{question}, 답변:{answer}"
        tokens = tokenizer.encode(full_text)
        
        eos_id = tokenizer.eos_id()
        if eos_id == -1: eos_id = 0
        tokens.append(eos_id)

        x = tokens[:-1]
        y = tokens[1:]

        pad_id = tokenizer.pad_id()
        if pad_id == -1: pad_id = 0

        x = x[:seq_len] + [pad_id] * (seq_len - len(x))
        y = y[:seq_len] + [pad_id] * (seq_len - len(y))
        
        return {"input_ids": x, "target_ids": y}

    # Filter out empty examples after tokenization
    dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names).filter(lambda x: len(x["input_ids"]) > 0)
    
    if args.num_train_examples is not None:
        dataset = dataset.take(args.num_train_examples)
        print(f"Using {args.num_train_examples} training examples.")

    return dataset


def load_model_and_state(args, tokenizer):
    """Initialize or load a model and its training state."""
    model_path = os.path.join(args.model_dir, "model.safetensors")
    state_path = os.path.join(args.model_dir, "training_state.json")

    # ModelArgs from gemma3.py
    # vocab_size is dynamically set from the tokenizer
    model_args = ModelArgs(vocab_size=tokenizer.get_piece_size(), model_type='gemma3')
    model = Model(model_args)
    optimizer = optim.Adam(learning_rate=args.lr)

    start_epoch = 0
    start_step = 0

    if not args.fresh_start and os.path.exists(model_path):
        print(f"Found existing checkpoint. Loading model from {model_path}...")
        try:
            raw_weights = mx.load(model_path)

            processed_weights = {}
            if isinstance(raw_weights, dict):
                for k, v in raw_weights.items():
                    if not isinstance(v, mx.array):
                        processed_weights[k] = mx.array(v)
                    else:
                        processed_weights[k] = v
            else:
                raise TypeError(f"Expected mx.load to return a dict, but got {type(raw_weights)}")

            sanitized_weights = model.sanitize(processed_weights)

            model_parameters = dict(model.parameters())
            for k, v in sanitized_weights.items():
                if k in model_parameters:
                    model_parameters[k].set(v)
                elif k.startswith("model.") and k[len("model."):] in model_parameters:
                    model_parameters[k[len("model."):]].set(v)
                elif k == "lm_head.weight" and "lm_head.weight" in model_parameters:
                    model_parameters["lm_head.weight"].set(v)
                elif k == "model.embed_tokens.weight" and "model.embed_tokens.weight" in model_parameters:
                    model_parameters["model.embed_tokens.weight"].set(v)
                elif k == "model.embed_tokens.weight" and "embed_tokens.weight" in model_parameters:
                    model_parameters["embed_tokens.weight"].set(v)
                else:
                    print(f"Warning: Parameter {k} not found in model.")

            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    start_epoch = state.get("epoch", 0)
                    start_step = state.get("step", 0)
                print(f"Resuming training from Epoch {start_epoch + 1}, Step {start_step + 1}")
            else:
                print("Warning: Model weights found, but no training state. Starting from beginning of epoch.")
        except Exception as e:
            print(f"Error loading weights: {e}. Starting from scratch.")
            start_epoch, start_step = 0, 0
    else:
        print("No checkpoint found or fresh start requested. Initializing new model.")

    mx.eval(model.parameters()) # Ensure parameters are initialized

    # Calculate and print model parameter count
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Model initialized with approximately {total_params / 1_000_000_000:.2f} Billion parameters.")

    return model, optimizer, start_epoch, start_step


def save_checkpoint(model, epoch, step, args):
    """Save model and training state."""
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.safetensors")
    state_path = os.path.join(args.model_dir, "training_state.json")

    print(f"\nSaving checkpoint at Epoch {epoch+1}, Step {step+1}...")
    save_safetensors(model_path, dict(tree_flatten(model.parameters())))
    
    with open(state_path, 'w') as f:
        json.dump({"epoch": epoch, "step": step}, f)
    print("Checkpoint saved.")


def train_step(model, optimizer, x, y, pad_token_id):
    # Ensure input is 2D for the model
    if x.ndim == 1:
        x = mx.expand_dims(x, axis=0)
        y = mx.expand_dims(y, axis=0)

    @mx.compile
    def loss_fn(params, x, y):
        """Calculate the loss, ignoring padding."""
        model.update(params)
        logits = model(x)
        logits = logits.astype(mx.float32)
        logits = logits.reshape(-1, model.args.vocab_size)
        y = y.reshape(-1)
        mask = (y != pad_token_id).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, y, reduction="none")
        masked_loss = loss * mask
        return mx.sum(masked_loss) / (mx.sum(mask) + 1e-9)

    loss, grads = mx.value_and_grad(loss_fn, argnums=0)(model.parameters(), x, y)
    optim.clip_grad_norm(grads, 1.0)
    optimizer.update(model, grads)
    return loss


def main():
    args = get_args()
    
    try:
        tokenizer = get_tokenizer(args.model_dir)
        print(f"Tokenizer loaded. Vocab size: {tokenizer.get_piece_size()}")
    except FileNotFoundError as e:
        print(e)
        return

    model, optimizer, start_epoch, start_step = load_model_and_state(args, tokenizer)
    dataset = load_dataset_and_preprocess(args, tokenizer, args.seq_len)
    if dataset is None:
        return

    pad_id = tokenizer.pad_id()
    if pad_id == -1: pad_id = 0

    print("\nStarting training...")
    
    # KorQuAD 2.1 has 83,486 training examples.
    num_examples = args.num_train_examples if args.num_train_examples is not None else 83486
    num_batches_per_epoch = num_examples // args.batch_size
    
    total_steps = args.epochs * num_batches_per_epoch
    warmup_steps = 100
    min_lr = 1e-7

    global_step = start_epoch * num_batches_per_epoch + start_step

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Shuffle the dataset for each epoch
        shuffled_dataset = dataset.shuffle() # Use a buffer for shuffling
        
        total_loss = 0
        
        pbar = tqdm(total=num_batches_per_epoch, desc=f"Epoch {epoch+1}")
        
        # Create an iterator for the shuffled dataset
        batch_iterator = iter(shuffled_dataset.iter(batch_size=args.batch_size))

        for step in range(num_batches_per_epoch):
            if global_step >= total_steps and args.max_steps == 0:
                break

            # Learning rate scheduling
            if global_step < warmup_steps:
                lr = args.lr * (global_step + 1) / warmup_steps
            else:
                cycle_steps = num_batches_per_epoch
                current_step_in_cycle = (global_step - warmup_steps) % cycle_steps
                progress = current_step_in_cycle / cycle_steps
                cosine_decay = 0.5 * (1 + mx.cos(mx.pi * progress))
                lr = min_lr + (args.lr - min_lr) * cosine_decay
            
            optimizer.learning_rate = lr

            try:
                batch = next(batch_iterator)
                x = mx.array(batch["input_ids"], dtype=mx.int32)
                y = mx.array(batch["target_ids"], dtype=mx.int32)
            except StopIteration:
                break # End of dataset

            loss = train_step(model, optimizer, x, y, pad_id)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6e}")
            pbar.update(1)
            global_step += 1

            if (global_step % args.save_every) == 0:
                save_checkpoint(model, epoch, global_step, args)
            
            if args.max_steps and global_step >= args.max_steps:
                break
        
        pbar.close()
        avg_loss = total_loss / (step + 1)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

        if args.max_steps and global_step >= args.max_steps:
            break

    print("\nTraining finished.")
    save_checkpoint(model, args.epochs - 1, global_step, args)


if __name__ == "__main__":
    main()