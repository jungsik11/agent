

import argparse
import mlx.core as mx
import sentencepiece as spm
import os
from gemma3 import ModelArgs, Model
from config import GEMMA3_ARGS, DATA_CACHE_DIR, checkpoint_dir as default_checkpoint_dir

def main(args):
    """
    Main function to run the interactive chat.
    """
    print("Loading tokenizer...")
    tokenizer_model_path = os.path.join(os.path.dirname(__file__), 'data', 'tokenizer.model')
    if not os.path.exists(tokenizer_model_path):
        print(f"Error: SentencePiece model file not found at {tokenizer_model_path}.")
        print("Please ensure the tokenizer exists. You might need to run prepare_dataset.py first.")
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

    print("\n--- Gemma3 Chat ---")
    print("모델이 준비되었습니다. 질문을 입력하세요.")
    print("종료하시려면 'exit' 또는 'quit'을 입력하세요.")
    print("--------------------------")

    max_tokens = 150  # Maximum number of tokens to generate

    while True:
        try:
            prompt = input("You: ")
        except EOFError:
            break
        if prompt.lower() in ["exit", "quit"]:
            break
        if not prompt.strip():
            continue

        tokens = sp_model.encode_as_ids(prompt)
        input_ids = mx.array([tokens])

        print("Gemma: ", end="", flush=True)
        
        response_tokens = []
        # The generate function is a generator, yielding one token at a time
        for token, count in zip(model.generate(input_ids, temp=args.temp, repetition_penalty=args.repetition_penalty), range(max_tokens)):
            token_id = token.item()
            
            # Stop generation if EOS token is produced
            if token_id == sp_model.eos_id():
                break
            
            response_tokens.append(token_id)
            
            # Decode and print the generated text so far
            # This provides a streaming-like effect
            current_text = sp_model.decode(response_tokens)
            # To avoid re-printing the whole text, we can find the new part,
            # but for simplicity, we'll just overwrite the line.
            print(f"\rGemma: {current_text}", end="", flush=True)

        print() # Move to the next line after generation is complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interact with a trained GeminiNano model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(default_checkpoint_dir, "model.safetensors"),
        help="Path to the trained model checkpoint (.safetensors file)."
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.5,
        help="The sampling temperature for generation."
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="The repetition penalty for generation."
    )
    args = parser.parse_args()
    main(args)

