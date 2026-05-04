import os
import sys
from pathlib import Path
import csv
from datetime import datetime
from zoneinfo import ZoneInfo


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from watermarking_llms import Watermark, WatermarkConfig

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, watermark: Watermark) -> None:
        self.watermark = watermark

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for batch_index in range(input_ids.shape[0]):
            green_token_ids = self.watermark.greenlist_tensor(
                input_ids[batch_index],
                device= scores.device
            )
            scores[batch_index, green_token_ids] += self.watermark.config.delta
        return scores


def run_watermark(prompt : str, model : str = "gpt2", max_new_tokens : int = 100, top_k : int = 50, 
                  gamma : float = 0.5, delta : float = 2.0, temperature : float = 0.8, analysis_token_step : int = 0) -> list[dict]:
    

    results = []

    device = pick_device()

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    watermark = Watermark(
        vocab_size=len(tokenizer),
        config=WatermarkConfig(gamma=gamma, delta=delta, hash_key=15_485_863),
        device=device
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
            logits_processor=LogitsProcessorList([WatermarkLogitsProcessor(watermark)]),
        )

    prompt_length = inputs["input_ids"].shape[1]
    continuation_ids = output[0, prompt_length:].tolist()
    generated_tokens = len(continuation_ids)


    if analysis_token_step == 0:
         analysis_token_step = generated_tokens

    #Given a step size [x], analyze the output generation for length x,2x,3x,...
    #Essentially, consider the first n tokens generated as a sequence of length n, for n = x,2x,3x,...
    #Significantly more efficient than generating a different sequence of each length
    for num_tokens in range(analysis_token_step,generated_tokens,analysis_token_step):
        scored_ids = output[0, prompt_length - 1 :prompt_length + num_tokens].tolist()
        analysis = watermark.detect(scored_ids)

        result = {
        "generated_tokens": num_tokens,
        "gamma": gamma,
        "delta": delta,
        "green_fraction": analysis.green_fraction,
        "z_score": analysis.z_score,
        "p_value": analysis.p_value,
        "prediction": analysis.prediction,
        }

        results.append(result)

    
    return results


def output_file_setup(outputfile : str):
    header = ["generated_tokens", "gamma", "delta", "green_fraction", "z_score", "p_value", "prediction"]

    with open(outputfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def write_results(outputfile : str, results : list[dict]):
    fields = ["generated_tokens", "gamma", "delta", "green_fraction", "z_score", "p_value", "prediction"]
    with open(outputfile, "a", newline="") as f:
        for result in results:
            row = [result[field] for field in fields]
            writer = csv.writer(f)
            writer.writerow(row)

