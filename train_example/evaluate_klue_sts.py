#!/usr/bin/env python
import argparse
import mlx.core as mx
import mlx.nn as nn
from scipy.stats import spearmanr
from datasets import load_dataset
import sentencepiece as spm
from tqdm import tqdm
import os

from gemma3 import ModelArgs, Model
from config import GEMMA3_ARGS, DATA_CACHE_DIR, checkpoint_dir as default_checkpoint_dir

def get_sentence_embedding(text: str, model: Model, sp_model: spm.SentencePieceProcessor) -> mx.array:
    """
    Generates a sentence embedding by taking the hidden state of the last token.
    """
    # Tokenize the input text
    tokens = sp_model.encode_as_ids(text)
    if not tokens:
        return mx.zeros((model.args.hidden_size,))
        
    input_ids = mx.array([tokens])

    # Get hidden states from the model
    hidden_states = model.model(input_ids)

    # Use the hidden state of the last token as the sentence embedding
    last_token_embedding = hidden_states[0, -1, :]
    
    return last_token_embedding

def main(args):
    print("Loading tokenizer...")
    tokenizer_model_path = os.path.join(os.path.dirname(__file__), 'data', 'tokenizer.model')
    if not os.path.exists(tokenizer_model_path):
        print(f"Error: SentencePiece model file not found at {tokenizer_model_path}.")
        return

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(tokenizer_model_path)
    vocab_size = sp_model.get_piece_size()
    print(f"SentencePiece tokenizer loaded. Vocabulary size: {vocab_size}")

    print("\nInitializing model...")
    model_args = ModelArgs(
        model_type=GEMMA3_ARGS["model_type"],
        hidden_size=GEMMA3_ARGS["hidden_size"],
        num_hidden_layers=GEMMA3_ARGS["num_hidden_layers"],
        intermediate_size=GEMMA3_ARGS["intermediate_size"],
        num_attention_heads=GEMMA3_ARGS["num_attention_heads"],
        head_dim=GEMMA3_ARGS["head_dim"],
        rms_norm_eps=GEMMA3_ARGS["rms_norm_eps"],
        num_key_value_heads=GEMMA3_ARGS["num_key_value_heads"],
        rope_traditional=GEMMA3_ARGS["rope_traditional"],
        vocab_size=vocab_size,
    )
    model = Model(model_args)

    print(f"Loading model weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    model.load_weights(args.model_path)
    model.eval()
    print("Model initialized and weights loaded.")

    print("\nLoading KLUE STS validation dataset...")
    try:
        dataset = load_dataset('klue', 'sts', split='validation')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    print(f"Dataset loaded. Number of examples: {len(dataset)}")

    predicted_scores = []
    gold_labels = []

    print("\nRunning evaluation...")
    for example in tqdm(dataset, desc="Evaluating KLUE STS"):
        sentence1 = example['sentence1']
        sentence2 = example['sentence2']
        gold_label = example['labels']['label']

        # Get embeddings for both sentences
        emb1 = get_sentence_embedding(sentence1, model, sp_model)
        emb2 = get_sentence_embedding(sentence2, model, sp_model)

        # Calculate cosine similarity
        # Normalize embeddings
        emb1_norm = emb1 / mx.linalg.norm(emb1)
        emb2_norm = emb2 / mx.linalg.norm(emb2)
        
        # Compute dot product for cosine similarity
        similarity = mx.sum(emb1_norm * emb2_norm).item()
        
        # Scale to 0-5 range (optional, but good for comparison)
        # Cosine similarity is in [-1, 1], we scale it to [0, 5]
        scaled_similarity = (similarity + 1) * 2.5

        predicted_scores.append(scaled_similarity)
        gold_labels.append(gold_label)

    # Calculate Spearman correlation
    correlation, p_value = spearmanr(predicted_scores, gold_labels)

    print("\n--- Evaluation Finished ---")
    print(f"KLUE STS Validation Set Performance:")
    print(f"Spearman Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print("---------------------------")
    print("\nNote: This is a zero-shot evaluation. Performance can be significantly improved with task-specific fine-tuning.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Gemma3 model on the KLUE STS benchmark.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(default_checkpoint_dir, "model.safetensors"),
        help="Path to the trained model checkpoint (.safetensors file)."
    )
    args = parser.parse_args()
    main(args)
