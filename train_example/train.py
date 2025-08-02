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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, "model_gemma3_3B")

    parser = argparse.ArgumentParser(description="Train a Gemma-3 model with MLX.")
    parser.add_argument("--model-dir", type=str, default=default_model_dir,
                        help="Directory to save/load model and tokenizer.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length for training.")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate.") # Further reduced LR
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--save-every", type=int, default=5, help="Save a checkpoint every N steps.")
    parser.add_argument("--fresh-start", action="store_true", default=True, # Set fresh-start to True by default
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
    """Download and preprocess the KorQuAD dataset."""
    print("Loading and preprocessing dataset...")
    try:
        dataset = load_dataset("kmgme/KorQuADv2_1", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    def strip_html_tags(text: str) -> str:
        """Removes HTML tags from a string."""
        return re.sub(re.compile('<.*?>'), '', text)

    def tokenize_fn(example):
        context = strip_html_tags(example.get('context', ''))
        question = example.get('question', '')
        answer = example.get('answers', {}).get('text', [''])[0]
        
        full_text = f"{context} {question} {answer}"
        tokens = tokenizer.encode(full_text)
        
        # Add EOS token at the end of the sequence
        eos_id = tokenizer.eos_id()
        if eos_id == -1: eos_id = 0 # Fallback if EOS token is not in vocab
        tokens.append(eos_id)

        # Input and target are the same, shifted by one
        x = tokens[:-1]
        y = tokens[1:]

        # Pad or truncate
        pad_id = tokenizer.pad_id()
        if pad_id == -1: pad_id = 0

        x = x[:seq_len] + [pad_id] * (seq_len - len(x))
        y = y[:seq_len] + [pad_id] * (seq_len - len(y))
        
        return {"input_ids": x, "target_ids": y}

    dataset = dataset.map(tokenize_fn, remove_columns=dataset.column_names)
    dataset.set_format(type="numpy")
    
    if args.num_train_examples is not None:
        dataset = dataset.select(range(args.num_train_examples))
        print(f"Using {len(dataset)} training examples.")

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
            model.load_weights(model_path)
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
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        num_batches = len(dataset) // args.batch_size
        total_loss = 0 # Initialize total loss for the epoch
        # Determine the starting step for the current epoch.
        # This is only relevant for the very first epoch of a resumed run.
        current_epoch_start_step = start_step if epoch == start_epoch else 0

        pbar = tqdm(range(current_epoch_start_step, num_batches), 
                    desc=f"Epoch {epoch+1}", 
                    initial=current_epoch_start_step, 
                    total=num_batches)
        
        for step in pbar:
            # Data loading
            start_idx = step * args.batch_size
            batch = dataset[start_idx : start_idx + args.batch_size]
            x = mx.array(batch["input_ids"])
            y = mx.array(batch["target_ids"])

            # Training step execution
            loss = train_step(model, optimizer, x, y, pad_id)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item() # Accumulate loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.refresh()
            global_step += 1

            # Save checkpoint
            if (step + 1) % args.save_every == 0:
                save_checkpoint(model, epoch, step, args)
            
            if args.max_steps and global_step >= args.max_steps:
                print(f"\nReached max steps ({args.max_steps}). Finishing training.")
                break
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

        # Reset start_step for subsequent epochs
        start_step = 0

        if args.max_steps and global_step >= args.max_steps:
            break

    print("\nTraining finished.")
    # Final model save
    save_checkpoint(model, args.epochs - 1, num_batches - 1, args)


if __name__ == "__main__":
    main()