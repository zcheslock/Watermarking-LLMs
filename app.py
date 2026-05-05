import csv
import json
import os
import random
import re
import sys
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watermarking_llms import Watermark, WatermarkConfig

SIMPLE_MODEL_NAME = "gpt2"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_CHOICES = [SIMPLE_MODEL_NAME, LLAMA_MODEL_NAME]
TEMPERATURE = 0.8
TOP_K = 50
DEFAULT_GAMMA = 0.5
DEFAULT_DELTA = 2.0
MODEL_MAX_NEW_TOKENS = {
    SIMPLE_MODEL_NAME: 80,
    LLAMA_MODEL_NAME: 40, # small, since running on M1 mac
}
RESULTS_DIR = ROOT / "results"
AGGREGATE_RESULTS = RESULTS_DIR / "game_results.csv"
# 20 claude generated prompts
PROMPT_BANK = [
    "A city after rain", "First day on Mars", "The last library",
    "A hidden doorway", "Breakfast at midnight", "The quiet inventor",
    "An empty stadium", "A storm at sea", "The lost notebook",
    "A small rebellion", "The future classroom", "An impossible map",
    "A train to nowhere", "The oldest tree", "A broken robot",
    "The moon colony", "An unexpected letter", "The final interview",
    "A museum at night", "The perfect disguise",
]

torch.set_num_threads(1)

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def model_dtype():
    if DEVICE in {"cuda", "mps"}:
        return torch.float16
    return None

def button(enabled: bool):
    return gr.update(interactive=enabled)

def score_text(score: int, attempts: int) -> str:
    if not attempts:
        return "Score: 0 / 0"
    return f"Score: {score} / {attempts} ({score / attempts:.0%})"

def settings_updates(enabled: bool):
    update = gr.update(interactive=enabled)
    return update, update, update, update, update, update

class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, watermark: Watermark) -> None:
        self.watermark = watermark

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for batch_index in range(input_ids.shape[0]):
            green_token_ids = self.watermark.greenlist_tensor(
                input_ids[batch_index], device=scores.device,
            )
            scores[batch_index, green_token_ids] += self.watermark.config.delta
        return scores

def sanitize_username(username: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", username.strip())
    return cleaned.strip("_") or "anon"

def upload_results(
    session_id: str,
    username: str,
    model_name: str,
    gamma: float,
    delta: float,
    round_results: dict[str, bool],
) -> str:
    score = sum(round_results.values())
    attempts = len(round_results)
    accuracy = score / attempts if attempts else 0.0
    RESULTS_DIR.mkdir(exist_ok=True)
    write_header = not AGGREGATE_RESULTS.exists()

    with AGGREGATE_RESULTS.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["timestamp", "session_id", "username", "model", "gamma", "delta", "round_results"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            session_id,
            sanitize_username(username),
            model_name,
            gamma,
            delta,
            json.dumps(round_results),
        ])

    return f"Results uploaded for {username}. Final accuracy: {score}/{attempts} ({accuracy:.0%})."

DEVICE = pick_device()
MODEL_BUNDLES: dict[str, tuple[AutoTokenizer, AutoModelForCausalLM]] = {}

def load_model_bundle(model_name: str):
    bundle = MODEL_BUNDLES.get(model_name)
    if bundle is not None:
        return bundle

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model_kwargs = {"local_files_only": True}
        dtype = model_dtype()
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(DEVICE)
    except Exception as error:
        raise RuntimeError(
            f"{model_name} is not downloaded locally yet. Download it first, then restart the app."
        ) from error

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    bundle = (tokenizer, model)
    MODEL_BUNDLES[model_name] = bundle
    return bundle

def watermark_processor(vocab_size: int, gamma: float, delta: float) -> LogitsProcessorList:
    watermark = Watermark(
        vocab_size=vocab_size,
        config=WatermarkConfig(gamma=gamma, delta=delta, hash_key=15_485_863),
    )
    return LogitsProcessorList([WatermarkLogitsProcessor(watermark)])

def generate_continuation(
    model_name: str,
    prompt: str,
    gamma: float,
    delta: float,
    *,
    watermarked: bool,
) -> str:
    tokenizer, model = load_model_bundle(model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    max_new_tokens = MODEL_MAX_NEW_TOKENS.get(model_name, 60)
    with torch.inference_mode():
        torch.manual_seed(random.randint(0, 1_000_000))
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=TOP_K,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
            logits_processor=watermark_processor(len(tokenizer), gamma, delta) if watermarked else LogitsProcessorList(),
        )
    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip()

def prepare_round(model_name: str, prompt: str, gamma: float, delta: float):
    watermarked_text = generate_continuation(model_name, prompt, gamma, delta, watermarked=True)
    plain_text = generate_continuation(model_name, prompt, gamma, delta, watermarked=False)
    if random.choice([True, False]):
        return watermarked_text, plain_text, "A"
    return plain_text, watermarked_text, "B"

def start_game(username: str, num_rounds: int):
    if not username.strip():
        return (
            "", "", None, [], {}, 0, "",
            "Please enter a username first.",
            0, 0, score_text(0, 0), button(False), button(False), button(False),
            *settings_updates(True),
            "",
        )

    prompts = random.sample(PROMPT_BANK, k=min(int(num_rounds), len(PROMPT_BANK)))
    return (
        "Generating...", "Generating...", None,
        prompts, {}, 0, prompts[0],
        f"Round 1 of {len(prompts)}. Generating continuations...",
        0, 0, score_text(0, 0), button(False), button(False), button(False),
        *settings_updates(False),
        str(uuid.uuid4()),
    )

def load_round(
    model_name: str,
    gamma: float,
    delta: float,
    prompts: list[str],
    round_index: int,
    current_status: str,
):
    if round_index >= len(prompts):
        return "", "", None, current_status, button(False), button(False)

    try:
        text_a, text_b, correct_answer = prepare_round(model_name, prompts[round_index], gamma, delta)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        return "", "", None, message, button(False), button(False)
    return (
        text_a, text_b, correct_answer,
        f"Round {round_index + 1} of {len(prompts)}. Which continuation is watermarked?",
        button(True),
        button(True),
    )

def handle_guess(
    guess: str,
    username: str,
    session_id: str,
    model_name: str,
    gamma: float,
    delta: float,
    prompts: list[str],
    round_results: dict[str, bool],
    round_index: int,
    correct_answer: str,
    score: int,
    attempts: int,
):
    attempts += 1
    was_correct = guess == correct_answer
    score += was_correct
    message = f"{'Correct' if was_correct else 'Incorrect'}. Text {correct_answer} was watermarked."

    updated_round_results = {**round_results, prompts[round_index]: was_correct}
    next_index = round_index + 1

    if next_index >= len(prompts):
        upload_message = upload_results(session_id, username, model_name, gamma, delta, updated_round_results)
        return (
            "", "", None, prompts, updated_round_results, next_index, "",
            f"{message} Game finished. {upload_message}",
            score, attempts, score_text(score, attempts), button(False), button(False), button(False),
            *settings_updates(True),
        )

    return (
        "Generating...", "Generating...", None,
        prompts, updated_round_results, next_index, prompts[next_index],
        f"{message} Generating round {next_index + 1} of {len(prompts)}...",
        score, attempts, score_text(score, attempts), button(False), button(False), button(True),
        *settings_updates(False),
    )

def upload_results_now(
    username: str,
    session_id: str,
    model_name: str,
    gamma: float,
    delta: float,
    round_results: dict[str, bool],
) -> str:
    if not username.strip():
        return "Please enter a username first."
    if not round_results:
        return "Play at least one round before uploading results."
    return upload_results(session_id, username, model_name, gamma, delta, round_results)

# UI
with gr.Blocks() as demo:
    gr.Markdown("# LLM Watermark Turing Test")

    session_id = gr.State("")
    prompts = gr.State([])
    round_results = gr.State({})
    round_index = gr.State(0)
    correct_answer = gr.State(None)
    score = gr.State(0)
    attempts = gr.State(0)

    username_input = gr.Textbox(label="Username", placeholder="your_name")
    model_input = gr.Dropdown(label="Model", choices=MODEL_CHOICES, value=SIMPLE_MODEL_NAME)
    num_rounds_input = gr.Dropdown(label="Rounds", choices=[1, 2, 3, 5, 10, 15, 20], value=5)
    gamma_input = gr.Slider(
        label="Gamma",
        minimum=0.1,
        maximum=0.9,
        step=0.05,
        value=DEFAULT_GAMMA,
        info="Fraction of the vocabulary placed in the green list.",
    )
    delta_input = gr.Slider(
        label="Delta",
        minimum=0.0,
        maximum=5.0,
        step=0.25,
        value=DEFAULT_DELTA,
        info="Amount we boost the probability of green tokens during generation.",
    )
    start_button = gr.Button("Start Game", variant="primary")
    prompt_display = gr.Textbox(label="Current Prompt", interactive=False)

    with gr.Row():
        text_a = gr.Textbox(label="Text A", lines=8, interactive=False)
        text_b = gr.Textbox(label="Text B", lines=8, interactive=False)

    with gr.Row():
        guess_a_button = gr.Button("Text A is Watermarked", interactive=False)
        guess_b_button = gr.Button("Text B is Watermarked", interactive=False)

    status = gr.Markdown("Waiting for the first round.")
    score_display = gr.Markdown("Score: 0 / 0")
    upload_button = gr.Button("Upload Results Now", interactive=False)

    game_outputs = [
        text_a,
        text_b,
        correct_answer,
        prompts,
        round_results,
        round_index,
        prompt_display,
        status,
        score,
        attempts,
        score_display,
        guess_a_button,
        guess_b_button,
        upload_button,
        username_input,
        model_input,
        num_rounds_input,
        gamma_input,
        delta_input,
        start_button,
    ]
    round_outputs = [text_a, text_b, correct_answer, status, guess_a_button, guess_b_button]
    guess_inputs = [
        username_input,
        session_id,
        model_input,
        gamma_input,
        delta_input,
        prompts,
        round_results,
        round_index,
        correct_answer,
        score,
        attempts,
    ]

    start_event = start_button.click(
        fn=start_game,
        inputs=[username_input, num_rounds_input],
        outputs=game_outputs + [session_id],
    )
    start_event.then(
        fn=load_round,
        inputs=[model_input, gamma_input, delta_input, prompts, round_index, status],
        outputs=round_outputs,
    )

    for guess_label, guess_button in [("A", guess_a_button), ("B", guess_b_button)]:
        guess_event = guess_button.click(
            fn=partial(handle_guess, guess_label),
            inputs=guess_inputs,
            outputs=game_outputs,
        )
        guess_event.then(
            fn=load_round,
            inputs=[model_input, gamma_input, delta_input, prompts, round_index, status],
            outputs=round_outputs,
        )

    upload_button.click(
        fn=upload_results_now,
        inputs=[username_input, session_id, model_input, gamma_input, delta_input, round_results],
        outputs=[status],
    )

if __name__ == "__main__":
    demo.launch(
        max_threads=1,
        theme=gr.themes.Monochrome(),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7861")),
        # share=True,
    )
