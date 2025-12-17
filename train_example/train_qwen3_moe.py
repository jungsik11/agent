import mlx.core as mx
from mlx.core import save_safetensors
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from datasets import load_from_disk
import json
import os
# import sentencepiece as spm # Removed
from tqdm import tqdm
import shutil
import re
from transformers import AutoTokenizer # Added

# Import ModelArgs and Model from qwen3_moe.py
from .qwen3_moe import ModelArgs, Model, QWEN3_MOE_1B_ARGS

# --- Configuration ---
MODEL_TYPE = "qwen3_moe"
MODEL_ARGS = QWEN3_MOE_1B_ARGS

# The directory where the model and tokenizer are stored.
# IMPORTANT: Make sure a compatible SentencePiece tokenizer model (tokenizer.model)
# is placed in this directory before running. For Qwen, a tokenizer from a model
# like 'Qwen/Qwen2-1.5B' would be appropriate.
CHECKPOINT_DIR = "train_example/qwen3_moe_1B_korean"

# Training parameters
batch_size = 4
seq_len = 128
learning_rate = 1e-5
num_epochs = 1
max_train_steps = 1 # Added for single step test

# --- Data Preprocessing ---
def tokenize_and_prepare_example(example, tokenizer, seq_len): # Changed sp_model to tokenizer
    """Tokenizes and formats a single example."""
    text = f"### 질문:\n{example['question']}\n\n### 답변:\n{example['answer']}"
    # input_ids = sp_model.encode(text) # Modified
    input_ids = tokenizer.encode(text, add_special_tokens=True) # Modified
    # pad_token_id = sp_model.pad_id() # Modified
    pad_token_id = tokenizer.pad_token_id # Modified
    if pad_token_id is None: # Added None check
        pad_token_id = 0 # Default to 0 if None
    # if pad_token_id == -1: # Removed
    #     pad_token_id = 0 # Removed
    
    target_ids = input_ids[1:] + [pad_token_id] # Keep target_ids creation similar but handle potential pad_token_id change
    input_ids = input_ids[:seq_len]
    target_ids = target_ids[:seq_len]
    while len(input_ids) < seq_len:
        input_ids.append(pad_token_id)
        target_ids.append(pad_token_id)
    return {"input_ids": input_ids, "target_ids": target_ids}

def save_tokenizer_files(source_dir, dest_dir):
    """Copies tokenizer files if they exist. (No longer needed for AutoTokenizer but keeping for now)"""
    pass # No longer needed, as AutoTokenizer handles downloading

# --- Training Step ---
def train_step(model, optimizer, x, y, tokenizer):
    """A single training step, including loss calculation and optimizer update."""
    def loss_fn(model, x, y, tokenizer):
        logits = model(x)
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)
        
        loss_per_token = nn.losses.cross_entropy(logits, y, reduction="none")
        
        # pad_token_id = tokenizer.pad_id() # Modified
        pad_token_id = tokenizer.pad_token_id # Modified
        if pad_token_id is None: # Added None check
            pad_token_id = 0 # Default to 0 if None
        # if pad_token_id == -1: # Removed
        #     pad_token_id = 0 # Removed
        
        mask = (y != pad_token_id).astype(mx.float32)
        
        masked_loss = loss_per_token * mask
        num_non_padding_tokens = mx.sum(mask)
        
        if num_non_padding_tokens > 0:
            return mx.sum(masked_loss) / num_non_padding_tokens
        else:
            return mx.array(0.0)

    loss, grads = mx.value_and_grad(loss_fn)(model, x, y, tokenizer)
    optimizer.update(model, grads)
    return loss

# --- Main Training Loop ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a Qwen3 MoE model.")
    parser.add_argument("--fresh-start", action="store_true", help="Start training from scratch, ignoring existing checkpoints.")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Load Tokenizer ---
    print("Loading tokenizer...")
    # tokenizer_path = os.path.join(CHECKPOINT_DIR, 'tokenizer.model') # Removed
    # if not os.path.exists(tokenizer_path): # Removed
    #     print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") # Removed
    #     print(f"!! IMPORTANT: tokenizer.model not found at {tokenizer_path}.") # Removed
    #     print(f"!! Please place a compatible SentencePiece tokenizer in that directory.") # Removed
    #     print(f"!! For Qwen models, you can get one from a Hugging Face repo like 'Qwen/Qwen2-1.5B'.") # Removed
    #     print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") # Removed
    #     return # Removed

    # tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path) # Modified
    try: # Added try-except for tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True) # Modified
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Check if tokenizer vocab size matches model args
    if tokenizer.vocab_size != MODEL_ARGS['vocab_size']: # Changed get_piece_size() to vocab_size
        print(f"Warning: Tokenizer vocab size ({tokenizer.vocab_size}) does not match model args vocab size ({MODEL_ARGS['vocab_size']}).")
        print("This may cause issues. Continuing anyway...")

    # --- Load Dataset ---
    print("Loading and preprocessing dataset...")
    try:
        dataset = load_from_disk("./data/korean_textbooks_qa")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenized_dataset = dataset.map(
        lambda example: tokenize_and_prepare_example(example, tokenizer, seq_len),
        remove_columns=['question', 'answer'],
        batched=False
    )
    tokenized_dataset.set_format(type="numpy", columns=["input_ids", "target_ids"])
    train_data = tokenized_dataset
    print(f"Dataset ready. Training examples: {len(train_data)}")

    # --- Initialize Model ---
    start_epoch = 0
    start_step = 0
    
    model = Model(ModelArgs(**MODEL_ARGS))
    mx.eval(model.parameters()) # Initialize parameters

    model_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
    state_path = os.path.join(CHECKPOINT_DIR, "training_state.json")

    if not args.fresh_start and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        try:
            model.load_weights(model_path)
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    training_state = json.load(f)
                    start_epoch = training_state.get("epoch", 0)
                    start_step = training_state.get("step", 0)
                print(f"Resuming training from Epoch {start_epoch + 1}, Step {start_step + 1}")
            else:
                print("No training state found. Starting new training session with pre-trained weights.")
        except (ValueError, KeyError, FileNotFoundError) as e:
            print(f"Error loading model weights: {e}")
            print("Starting training from scratch as weights could not be loaded or are incompatible.")
            start_epoch, start_step = 0, 0
    else:
        print("No pre-trained model found or --fresh-start used. Training from scratch.")

    # --- Start Training ---
    optimizer = optim.Adam(learning_rate=learning_rate)
    num_batches = len(train_data) // batch_size
    
    print("Starting training...\n")
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        indices = mx.random.permutation(mx.array(mx.arange(len(train_data))))
        
        first_step_to_process = 0
        if epoch == start_epoch and start_step > 0:
            first_step_to_process = start_step

        # Limit the range for tqdm to max_train_steps
        pbar_range = range(first_step_to_process, min(num_batches, first_step_to_process + max_train_steps))
        pbar = tqdm(pbar_range, desc=f"Epoch {epoch + 1}", initial=first_step_to_process, total=min(num_batches, first_step_to_process + max_train_steps))


        for current_step in pbar:
            if current_step >= first_step_to_process + max_train_steps: # Ensure we don't exceed max_train_steps
                break
            i = current_step * batch_size
            batch_indices = indices[i : i + batch_size].tolist()
            x = mx.array(train_data[batch_indices]["input_ids"])
            y = mx.array(train_data[batch_indices]["target_ids"])

            loss = train_step(model, optimizer, x, y, tokenizer)
            mx.eval(model.parameters(), optimizer.state)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Save checkpoint if running for more than one step
            if max_train_steps > 1 and current_step > 0 and current_step % 100 == 0:
                save_tokenizer_files(CHECKPOINT_DIR, CHECKPOINT_DIR)
                save_safetensors(os.path.join(CHECKPOINT_DIR, "model.safetensors"), dict(tree_flatten(model.parameters())))
                with open(state_path, 'w') as f:
                    json.dump({'epoch': epoch, 'step': current_step}, f)
        
        start_step = 0 # Reset for next epoch
        
        print(f"Epoch {epoch + 1} finished.")

        # Save final checkpoint
        save_tokenizer_files(CHECKPOINT_DIR, CHECKPOINT_DIR)
        save_safetensors(os.path.join(CHECKPOINT_DIR, "model.safetensors"), dict(tree_flatten(model.parameters())))
        with open(state_path, 'w') as f:
            json.dump({'epoch': epoch + 1, 'step': 0}, f)

if __name__ == "__main__":
    main()