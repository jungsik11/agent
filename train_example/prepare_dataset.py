import os
import json
import sentencepiece as spm
from mlx_lm import load

DATA_CACHE_DIR = "data"
HUGGING_FACE_REPO = "mlx-community/Llama-4-Maverick-17B-16E-Instruct-4bit"

def setup_tokenizer():
    """
    Loads the tokenizer using mlx_lm and saves it to the data directory.
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    print(f"Loading tokenizer for: {HUGGING_FACE_REPO}")
    print("This may download model parts, but only tokenizer will be saved.")

    try:
        # Load the model and tokenizer from Hugging Face
        # We only need the tokenizer, but load() returns both
        _, tokenizer = load(HUGGING_FACE_REPO)
        
        # Save the tokenizer to the data directory
        tokenizer.save_pretrained(DATA_CACHE_DIR)
        print(f"Tokenizer saved to: {DATA_CACHE_DIR}")

        # Verify the tokenizer.model file exists and get vocab size
        tokenizer_model_path = os.path.join(DATA_CACHE_DIR, "tokenizer.model")
        if os.path.exists(tokenizer_model_path):
            sp_model = spm.SentencePieceProcessor()
            sp_model.load(tokenizer_model_path)
            vocab_size = sp_model.get_piece_size()
            print(f"Vocabulary Size: {vocab_size}")
        else:
            print("Warning: tokenizer.model not found after saving. Check mlx_lm tokenizer structure.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Failed to load or save the tokenizer using mlx_lm.")
        print("Please ensure you have mlx_lm installed and an internet connection.")
        return

if __name__ == "__main__":
    setup_tokenizer()