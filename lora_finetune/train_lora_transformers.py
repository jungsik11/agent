import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA using Transformers.")
    parser.add_argument("--model_id", type=str, default="beomi/gemma-ko-2b", help="The model ID from Hugging Face.")
    parser.add_argument("--dataset_id", type=str, default="HAERAE-HUB/KOREAN-WEBTEXT", help="The dataset ID from Hugging Face.")
    parser.add_argument("--output_dir", type=str, default="lora_finetune/gemma-ko-webtext-adapter-transformers", help="Directory to save the LoRA adapter.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length.")

    args = parser.parse_args()

    # --- 1. Device Setup ---
    # With PyTorch 2.0 and later, MPS device can be used directly
    # The `use_mps_device` flag in TrainingArguments is deprecated, so we manage device placement manually.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # --- 2. Load Model and Tokenizer ---
    print(f"Loading model: {args.model_id}")
    # Using bfloat16 for better performance, requires appropriate hardware
    # On MPS, float32 is often more stable, but let's try bfloat16 first
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
        )
    except TypeError:
        print("bfloat16 not supported, falling back to float32.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float32,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to the detected device
    model.to(device)
    
    # --- 3. LoRA Configuration (PEFT) ---
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print("LoRA model configured. Trainable parameters:")
    print_trainable_parameters(model)

    # --- 4. Load and Process Dataset ---
    print(f"Loading and processing dataset: {args.dataset_id}")
    dataset = load_dataset(args.dataset_id)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_seq_length, padding="max_length", return_tensors="pt")

    # Use a smaller subset for quick testing if needed
    # raw_dataset = dataset["train"].select(range(10000)) 
    raw_dataset = dataset["train"]

    tokenized_datasets = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)
    
    split_dataset = tokenized_datasets.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Training on {len(train_dataset)} examples, evaluating on {len(eval_dataset)} examples.")

    # --- 5. Trainer Setup (Using user's provided example) ---
    print("Setting up Trainer...")
    # Correcting typo from 'loggint_steps' to 'logging_steps'
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_steps=1000,
        eval_steps=500, # Re-adding eval_steps
        save_steps=500, # Re-adding save_steps
        save_total_limit=3,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        fp16=False,  # fp16 is not recommended for MPS, bfloat16 is preferred or float32
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # The model is already on the correct device, so Trainer will use it.

    # --- 6. Start Training ---
    print("Starting training...")
    trainer.train()

    # --- 7. Save Final Adapter ---
    print("Training finished. Saving final LoRA adapter.")
    model.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()