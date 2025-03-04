

from mlx_lm import load, generate
import sys
import argparse


def build_parser():
    parser = argparse.ArgumentParser(description="generate text")
    parser.add_argument(
        "--text",
        type=str,
        help="The input text of a llm model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the local model directory.",
    )
    return parser


def run_llm(model, text):

    model, tokenizer = load(model)

    messages= f"<|user|>\n{text} <|end|>"

    response = generate(model, tokenizer, prompt=messages, verbose=False, max_tokens=1024)

    return response

def main():
    parser = build_parser()
    args = parser.parse_args()
    response = run_llm(args.model, args.text)
    return response


if __name__ == "__main__":
    out = main()
    print(out)


