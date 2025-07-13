import mlx.core as mx
from mlx.core import save_safetensors
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from datasets import load_dataset
import json
import os
from dataclasses import dataclass
import sentencepiece as spm
from tqdm import tqdm # Import tqdm
import shutil
import re

# Import ModelArgs and GeminiNano from gemini_nano.py
from gemma3 import ModelArgs, Model

# Load config values from config.py
from config import (
    batch_size,
    learning_rate,
    num_epochs,
    seq_len,
    checkpoint_dir,
    DATA_CACHE_DIR,
    korquad_data_dir,
    MODEL_TYPE,
    GEMINI_NANO_ARGS,
    GEMMA3_ARGS,
)

# Data Preprocessing Function
def strip_html_tags(text):
    """Removes HTML tags from a string."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def tokenize_and_prepare_example(example, sp_model, seq_len):
    # KorQuAD dataset structure when loaded directly from Hugging Face:
    # example has keys: 'id', 'title', 'context', 'question', 'answers'
    # 'answers' is a dictionary with 'text' (list) and 'answer_start' (list)

    context = strip_html_tags(example['context']) # Strip HTML tags
    question = example['question']
    answer = ""
    if 'answers' in example and example['answers'] and 'text' in example['answers'] and example['answers']['text']:
        answer = example['answers']['text'][0]

    # Concatenate context, question, and answer
    full_text = context + " " + question + " " + answer

    # Convert text to IDs using the SentencePiece tokenizer
    input_ids = sp_model.encode(full_text)

    # Create target IDs (shifted input_ids)
    pad_token_id = sp_model.pad_id()
    if pad_token_id == -1:
        pad_token_id = 0  # Fallback if PAD token is not in vocab
    target_ids = input_ids[1:] + [pad_token_id]

    # Truncate or pad to seq_len
    input_ids = input_ids[:seq_len]
    target_ids = target_ids[:seq_len]

    while len(input_ids) < seq_len:
        input_ids.append(pad_token_id)
        target_ids.append(pad_token_id)

    return {"input_ids": input_ids, "target_ids": target_ids}

def save_tokenizer_files(checkpoint_dir):
    """Saves tokenizer files to the specified directory."""
    tokenizer_model_src = os.path.join(DATA_CACHE_DIR, 'tokenizer.model')
    tokenizer_vocab_src = os.path.join(DATA_CACHE_DIR, 'tokenizer.vocab')
    
    tokenizer_model_dest = os.path.join(checkpoint_dir, 'tokenizer.model')
    tokenizer_vocab_dest = os.path.join(checkpoint_dir, 'tokenizer.vocab')

    if os.path.exists(tokenizer_model_src) and os.path.exists(tokenizer_vocab_src):
        shutil.copy(tokenizer_model_src, tokenizer_model_dest)
        shutil.copy(tokenizer_vocab_src, tokenizer_vocab_dest)
    else:
        pass

# Training function
def train_step(model, optimizer, x, y, tokenizer):
    def loss_fn(model, x, y, tokenizer):
        logits = model(x)
        # Reshape for cross_entropy: (batch_size * seq_len, vocab_size)
        # y: (batch_size * seq_len)
        logits = logits.reshape(-1, logits.shape[-1])
        y = y.reshape(-1)

        # Calculate cross-entropy loss for all tokens
        loss_per_token = nn.losses.cross_entropy(logits, y, reduction="none")

        # Create a mask to ignore padding tokens
        pad_token_id = tokenizer.pad_id()
        if pad_token_id == -1:
            pad_token_id = 0 # Fallback
        mask = (y != pad_token_id).astype(mx.float32)

        # Apply the mask to the loss
        masked_loss = loss_per_token * mask

        # Calculate the mean loss only over non-padding tokens
        # Avoid division by zero if there are no non-padding tokens
        num_non_padding_tokens = mx.sum(mask)
        if num_non_padding_tokens > 0:
            return mx.sum(masked_loss) / num_non_padding_tokens
        else:
            return mx.array(0.0) # Return 0 loss if no non-padding tokens

    loss, grads = mx.value_and_grad(loss_fn)(model, x, y, tokenizer)
    optimizer.update(model, grads)
    return loss

# Main training loop
def main():
    print("Loading tokenizer...")
    tokenizer_path = os.path.join(DATA_CACHE_DIR, 'tokenizer.model')
    if not os.path.exists(tokenizer_path):
        print(f"Error: tokenizer.model not found at {tokenizer_path}. Please run prepare_dataset.py first.")
        return

    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    vocab_size = tokenizer.get_piece_size()
    print(f"Tokenizer loaded from json. Vocabulary size: {vocab_size}")

    print("Loading dataset...")
    try:
        # Load directly from Hugging Face Hub
        dataset = load_dataset("kmgme/KorQuADv2_1")
    except Exception as e:
        print(f"Error loading dataset: {e}. Please ensure you have access to kmgme/KorQuADv2_1 on Hugging Face Hub.")
        return

    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        lambda example: tokenize_and_prepare_example(example, tokenizer, seq_len),
        remove_columns=dataset['train'].column_names,
        batched=False  # Process one example at a time
    )
    tokenized_dataset.set_format(type="numpy", columns=["input_ids", "target_ids"])

    train_data = tokenized_dataset["train"].select(range(10000))
    # validation_data = tokenized_dataset["validation"] # Not using validation for this simple script
    print(f"Dataset preprocessed. Training examples: {len(train_data)}")

    # Import model based on MODEL_TYPE
    if MODEL_TYPE == "gemini_nano":
        from gemini_nano import ModelArgs, GeminiNano as ModelClass
        model_args_dict = GEMINI_NANO_ARGS
    elif MODEL_TYPE == "gemma3":
        from gemma3 import ModelArgs, Model as ModelClass
        model_args_dict = GEMMA3_ARGS
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    # Model initialization
    print("Initializing model...\n")
    model_args = ModelArgs(vocab_size=vocab_size, **model_args_dict)
    model = ModelClass(model_args)
    mx.eval(model.parameters())  # Initialize parameters
    print("Model initialized.\n")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=learning_rate)

    # Initialize start_epoch and start_step
    start_epoch = 0
    start_step = 0

    # Check for existing model and training state
    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    state_path = os.path.join(checkpoint_dir, "training_state.json")

    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        model.load_weights(model_path)
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
                start_epoch = state.get('epoch', 0)
                start_step = state.get('step', 0)
            print(f"Resuming training from Epoch {start_epoch + 1}, Step {start_step + 1}")
        else:
            print("No training state found. Starting from Epoch 1, Step 1.")
    else:
        print("No existing model found. Starting from scratch.")


    # Training loop
    print("Starting training...\n")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Shuffle and batch data
        indices = mx.array(mx.arange(len(train_data)))
        indices = mx.random.permutation(indices)

        total_loss = 0
        
        # Determine the starting step for the current epoch.
        # This is only relevant for the very first epoch of a resumed run.
        first_step_to_process = 0
        if epoch == start_epoch and start_step > 0:
            # Resume from the step *after* the last completed one.
            first_step_to_process = start_step + 1

        # Setup progress bar
        num_batches = len(train_data) // batch_size
        pbar = tqdm(range(first_step_to_process, num_batches), desc=f"Epoch {epoch + 1}", initial=first_step_to_process, total=num_batches)

        # Batch loop
        for current_step in pbar:
            i = current_step * batch_size
            batch_indices = indices[i : i + batch_size]
            batch_indices_list = batch_indices.tolist()
            x = mx.array(train_data[batch_indices_list]["input_ids"])
            y = mx.array(train_data[batch_indices_list]["target_ids"])

            loss = train_step(model, optimizer, x, y, tokenizer)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Save checkpoint every 5 steps
            if current_step > 0 and current_step % 5 == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
                save_safetensors(checkpoint_path, dict(tree_flatten(model.parameters())))
                save_tokenizer_files(checkpoint_dir)
                
                # Save training state
                state_path = os.path.join(checkpoint_dir, "training_state.json")
                with open(state_path, 'w') as f:
                    json.dump({'epoch': epoch, 'step': current_step}, f)
        
        # Reset start_step after the first epoch
        start_step = 0
            
        # Calculate average loss using the total number of batches from tqdm
        avg_loss = total_loss / (len(train_data) / batch_size) # Use actual number of batches
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

        # Save checkpoint at the end of the epoch
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "model.safetensors")
        save_safetensors(checkpoint_path, dict(tree_flatten(model.parameters())))

        save_tokenizer_files(checkpoint_dir)

        # Save training state
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        with open(state_path, 'w') as f:
            json.dump({'epoch': epoch + 1, 'step': 0}, f) # Save next epoch, step 0

if __name__ == "__main__":
    main()