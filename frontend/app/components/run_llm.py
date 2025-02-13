

from mlx_lm import load, generate

import argparse

model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'



def build_parser():
    parser = argparse.ArgumentParser(description="generate text")
    parser.add_argument( "--text", type=str, default='there is no text' )
    return parser


def run_llm(input_text =None):

    
    #parser = build_parser()
    #args = parser.parse_args()
    model, tokenizer = load(model_id)

    messages= f"<|user|>\n{input_text} <|end|>"

    response = generate(model, tokenizer, prompt=messages, verbose=False)

    return response

if __name__ == "__main__":
    
    out = run_llm()
    print(out)


