import argparse
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analysis import (
    WatermarkLogitsProcessor,
    continuation_perplexity,
    continuation_token_losses,
    pick_device,
)
from watermarking_llms import Watermark, WatermarkConfig


MODEL_NAME = "gpt2"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50
DEFAULT_GAMMA = 0.5
DEFAULT_DELTA = 2.0

torch.set_num_threads(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate watermarked text from a Hugging Face model.")
    parser.add_argument("prompt", help="Prompt text to continue.")
    parser.add_argument("--model", default=MODEL_NAME, help=f"Hugging Face generation model. Default: {MODEL_NAME}")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Number of tokens to generate. Default: {MAX_NEW_TOKENS}",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help=f"Fraction of the vocabulary in the green list. Default: {DEFAULT_GAMMA}",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DEFAULT_DELTA,
        help=f"Boost applied to green token logits. Default: {DEFAULT_DELTA}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    watermark = Watermark(
        vocab_size=len(tokenizer),
        config=WatermarkConfig(gamma=args.gamma, delta=args.delta, hash_key=15_485_863),
        device=device,
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_k=TOP_K,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
            logits_processor=LogitsProcessorList([WatermarkLogitsProcessor(watermark)]),
        )

    prompt_length = inputs["input_ids"].shape[1]
    continuation_ids = output[0, prompt_length:].tolist()
    scored_ids = output[0, prompt_length - 1 :].tolist()
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    token_losses = continuation_token_losses(model, output, prompt_length)

    analysis = watermark.detect(scored_ids)
    perplexity = continuation_perplexity(token_losses, len(continuation_ids))

    print(generated_text)
    print()
    print("Watermark analysis")
    print(f"Generation model: {args.model}")
    print(f"Generated tokens: {len(continuation_ids)}")
    print(f"Perplexity: {perplexity:.3f}")
    print(f"Green fraction: {analysis.green_fraction:.3f}")
    print(f"Green percentage: {100 * analysis.green_fraction:.1f}%")
    print(f"Z-score: {analysis.z_score:.3f}")
    print(f"P-value: {analysis.p_value:.3g}")
    print(f"Prediction: {'watermarked' if analysis.prediction else 'not watermarked'}")


if __name__ == "__main__":
    main()
