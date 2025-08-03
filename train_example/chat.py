import argparse
import mlx.core as mx
import sentencepiece as spm
import os
from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from gemma3 import ModelArgs, Model

class Gemma3LLM(LLM):
    model: Any
    tokenizer: Any
    model_args: Any
    temp: float
    repetition_penalty: float

    @property
    def _llm_type(self) -> str:
        return "gemma3"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        tokens = self.tokenizer.encode_as_ids(prompt)
        input_ids = mx.array([tokens])

        generated_tokens = []
        # Max tokens hardcoded for now, can be an arg
        for token, _ in zip(self.model.generate(input_ids, temp=self.temp, repetition_penalty=self.repetition_penalty), range(150)):
            token_id = token.item()
            if token_id == self.tokenizer.eos_id():
                break
            generated_tokens.append(token_id)

        return self.tokenizer.decode(generated_tokens)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "temp": self.temp,
            "repetition_penalty": self.repetition_penalty,
        }

def main(args):
    """
    Main function to run the interactive chat.
    """
    print("Loading tokenizer...")
    tokenizer_model_path = os.path.join(os.path.dirname(args.model_path), 'tokenizer.model')
    if not os.path.exists(tokenizer_model_path):
        print(f"Error: SentencePiece model file not found at {tokenizer_model_path}.")
        print("Please ensure the tokenizer exists. You might need to run prepare_dataset.py first.")
        return

    sp_model = spm.SentencePieceProcessor()
    sp_model.load(tokenizer_model_path)
    vocab_size = sp_model.get_piece_size()
    print(f"SentencePiece tokenizer loaded. Vocabulary size: {vocab_size}")

    print("\nInitializing model...")
    # Use default ModelArgs from gemma3.py, only overriding vocab_size
    model_args = ModelArgs(vocab_size=vocab_size, model_type='gemma3')
    model = Model(model_args)

    print(f"Loading model weights from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    model.load_weights(args.model_path)
    model.eval()
    print("Model initialized and weights loaded.")

    # Initialize Gemma3LLM
    llm = Gemma3LLM(
        model=model,
        tokenizer=sp_model,
        model_args=model_args,
        temp=args.temp,
        repetition_penalty=args.repetition_penalty,
    )

    # Define PromptTemplate
    template = """You are a helpful AI assistant. Answer the following question:

    Question: {query}
    Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["query"])

    # Create LLMChain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    print("\n--- Gemma3 Chat ---")
    print("모델이 준비되었습니다. 질문을 입력하세요.")
    print("--------------------------")

    query = args.query
    if not query.strip():
        print("Error: No query provided.")
        return

    print("Gemma: ", end="", flush=True)
    response = llm_chain.run(query)
    print(response)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interact with a trained Gemma3 model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("./model_gemma3", "model.safetensors"),
        help="Path to the trained model checkpoint (.safetensors file).",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.5,
        help="The sampling temperature for generation.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="The repetition penalty for generation.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="The query to ask the model.",
    )
    args = parser.parse_args()
    main(args)