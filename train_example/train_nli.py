import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
import dataclasses

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sentencepiece as spm
from datasets import load_dataset, load_from_disk
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
    parser.add_argument("--lr-patience", type=int, default=20, help="Patience in steps for LR scheduler.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Factor to reduce LR by.")
    parser.add_argument("--min-lr", type=float, default=1e-7, help="Minimum learning rate.")
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

        # Remove "**Phi:**" prefix from the answer
        answer = answer.replace('**Phi:**', '').strip()
        
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
    optimizer_path = os.path.join(args.model_dir, "optimizer.npz")

    start_epoch = 0
    start_step = 0
    best_loss = float('inf')
    last_avg_loss = float('inf')
    state = {}

    # First, determine the starting step
    if not args.fresh_start and os.path.exists(model_path) and os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
            start_epoch = state.get("epoch", 0)
            start_step = state.get("step", 0)
            best_loss = state.get("best_loss", float('inf'))
            last_avg_loss = state.get("last_avg_loss", float('inf'))

    # Per user request, do not use config.json. Using default ModelArgs from gemma3.py
    print("Ignoring config.json and using default model arguments.")
    model_args = ModelArgs(model_type="gemma3", vocab_size=tokenizer.get_piece_size())
    model = Model(model_args)
    mx.eval(model.parameters()) # Ensure parameters are initialized
    optimizer = optim.Adam(learning_rate=args.lr)

    # Load model weights and optimizer state if resuming
    if start_step > 0:
        print(f"Found existing checkpoint. Loading model from {model_path}...")
        try:
            # Load model weights
            weights = mx.load(model_path)
            weights = model.sanitize(weights)
            model.update(weights)

            # Load optimizer state
            if os.path.exists(optimizer_path):
                optimizer.update_from_numpy(str(optimizer_path))
                print("Optimizer state loaded.")

            # Restore learning rate from state file (can override optimizer state)
            saved_lr = state.get("lr", args.lr)
            optimizer.learning_rate = saved_lr

            print(f"Resuming training from Epoch {start_epoch + 1}, Step {start_step + 1}")
            print(f"Restored learning rate to {saved_lr:.6e}")
            print(f"Restored best loss to {best_loss:.4f}")
            print(f"Restored last average loss to {last_avg_loss:.4f}")

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Treating as a fresh start.")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            start_epoch, start_step = 0, 0 # Reset if weights fail to load
    else:
        print("No checkpoint found or fresh start requested. Initializing new model.")

    # Calculate and print model parameter count
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Model initialized with approximately {total_params / 1_000_000_000:.2f} Billion parameters.")

    return model, optimizer, start_epoch, start_step, best_loss, last_avg_loss


def save_checkpoint(model, optimizer, epoch, step, best_loss, last_avg_loss, args):
    """Save model and training state."""
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.safetensors")
    state_path = os.path.join(args.model_dir, "training_state.json")
    optimizer_path = os.path.join(args.model_dir, "optimizer.npz")

    print(f"\nSaving checkpoint at Epoch {epoch+1}, Step {step+1} with best_loss {best_loss:.4f}...")
    save_safetensors(model_path, dict(tree_flatten(model.parameters())))
    mx.savez(optimizer_path, **optimizer.state)
    
    with open(state_path, 'w') as f:
        json.dump({
            "epoch": epoch, 
            "step": step, 
            "lr": float(optimizer.learning_rate), 
            "best_loss": best_loss,
            "last_avg_loss": last_avg_loss
        }, f)
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

    model, optimizer, start_epoch, start_step, best_loss, last_avg_loss = load_model_and_state(args, tokenizer)
    dataset = load_dataset_and_preprocess(args, tokenizer, args.seq_len)
    if dataset is None:
        return

    pad_id = tokenizer.pad_id()
    if pad_id == -1: pad_id = 0

    print("\nStarting training...")

    num_examples = args.num_train_examples if args.num_train_examples is not None else 83486
    num_batches_per_epoch = num_examples // args.batch_size

    global_step = start_step
    
    # Separate loss windows for saving and LR scheduling
    save_loss_window = []
    lr_loss_window = []

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        shuffled_dataset = dataset.shuffle()
        batch_iterator = iter(shuffled_dataset.iter(batch_size=args.batch_size))

        steps_already_done = 0
        if epoch == start_epoch and global_step > 0:
            steps_already_done = global_step % num_batches_per_epoch
            if steps_already_done > 0:
                print(f"Resuming from global step {global_step}. Skipping {steps_already_done} batches in this epoch.")
                for _ in range(steps_already_done):
                    try:
                        next(batch_iterator)
                    except StopIteration:
                        print("Warning: Reached end of dataset while skipping batches.")
                        break
        
        total_loss = 0
        pbar = tqdm(initial=steps_already_done, total=num_batches_per_epoch, desc=f"Epoch {epoch+1}")

        for step in range(steps_already_done, num_batches_per_epoch):
            if args.max_steps and global_step >= args.max_steps:
                break

            try:
                batch = next(batch_iterator)
                x = mx.array(batch["input_ids"], dtype=mx.int32)
                y = mx.array(batch["target_ids"], dtype=mx.int32)
            except StopIteration:
                break

            loss = train_step(model, optimizer, x, y, pad_id)
            mx.eval(model.parameters(), optimizer.state)

            current_loss = loss.item()
            total_loss += current_loss
            save_loss_window.append(current_loss)
            lr_loss_window.append(current_loss)

            pbar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{optimizer.learning_rate:.6e}")
            pbar.update(1)
            global_step += 1

            # Conditional saving block (uses save_every)
            if (global_step % args.save_every) == 0 and global_step > 0:
                if save_loss_window:
                    current_avg_loss = sum(save_loss_window) / len(save_loss_window)
                    if current_avg_loss < best_loss:
                        print(f"\nStep {global_step}: Avg loss {current_avg_loss:.4f} is better than best {best_loss:.4f}. Saving checkpoint.")
                        best_loss = current_avg_loss
                        save_checkpoint(model, optimizer, epoch, global_step, best_loss, last_avg_loss, args)
                    else:
                        print(f"\nStep {global_step}: Avg loss {current_avg_loss:.4f} is not better than best {best_loss:.4f}. Skipping save.")
                    save_loss_window = [] # Reset window after check

            # Learning rate scheduler block (uses lr_patience)
            if (global_step % args.lr_patience) == 0 and global_step > 0:
                if lr_loss_window:
                    current_lr_avg_loss = sum(lr_loss_window) / len(lr_loss_window)
                    if current_lr_avg_loss >= last_avg_loss:
                        current_lr = optimizer.learning_rate
                        new_lr = current_lr * args.lr_factor
                        if new_lr >= args.min_lr:
                            optimizer.learning_rate = new_lr
                            print(f"\nStep {global_step}: Reducing learning rate from {current_lr:.6e} to {new_lr:.6e}")
                        else:
                            optimizer.learning_rate = args.min_lr
                            print(f"\nStep {global_step}: Learning rate reached minimum of {args.min_lr:.6e}")
                    last_avg_loss = current_lr_avg_loss
                    lr_loss_window = [] # Reset window after check

            if args.max_steps and global_step >= args.max_steps:
                break
        
        pbar.close()
        avg_loss = total_loss / (step + 1) if step > 0 else total_loss
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

        if args.max_steps and global_step >= args.max_steps:
            break

    print("\nTraining finished.")
    # Final save based on the last window of losses
    final_loss = (sum(save_loss_window) / len(save_loss_window)) if save_loss_window else float('inf')
    if final_loss < best_loss:
        print(f"Final loss {final_loss:.4f} is better than best loss {best_loss:.4f}. Saving final model.")
        best_loss = final_loss
    else:
        print(f"Final loss {final_loss:.4f} is not better than best loss {best_loss:.4f}. Saving with last best loss.")
    save_checkpoint(model, optimizer, args.epochs - 1, global_step, best_loss, last_avg_loss, args)


if __name__ == "__main__":
    main()