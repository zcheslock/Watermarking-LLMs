from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import exp, sqrt
from random import Random
from statistics import NormalDist
from typing import Callable, Sequence

import torch

TokenLogitsFn = Callable[[Sequence[int]], Sequence[float]]

@dataclass(frozen=True)
class WatermarkConfig:
    gamma: float = 0.5
    delta: float = 2.0
    hash_key: int = 15_485_863
    z_threshold: float = 4.0
    select_green_tokens: bool = True

@dataclass(frozen=True)
class DetectionResult:
    num_tokens_scored: int
    num_green_tokens: int
    green_fraction: float
    z_score: float
    p_value: float
    prediction: bool
    green_token_mask: list[bool] | None = None

def sample_from_logits(logits: Sequence[float], rng: Random) -> int:
    max_logit = max(logits)
    weights = [exp(logit - max_logit) for logit in logits]
    total = sum(weights)
    draw = rng.random() * total

    running_total = 0.0
    for token_id, weight in enumerate(weights):
        running_total += weight
        if draw <= running_total:
            return token_id

    return len(weights) - 1


class Watermark:
    def __init__(self, vocab_size: int, config: WatermarkConfig | None = None, device: str = "cpu") -> None:
        self.vocab_size = vocab_size
        self.config = config or WatermarkConfig()
        self.device = device

    def greenlist_size(self) -> int:
        return int(self.vocab_size * self.config.gamma)

    def _seed_value(self, prefix_tokens: Sequence[int]) -> int:
        return self.config.hash_key * int(prefix_tokens[-1])

    def greenlist_tensor(
        self,
        prefix_tokens: Sequence[int] | torch.LongTensor,
        *,
        device: str | torch.device,
    ) -> torch.LongTensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(self._seed_value(prefix_tokens))
        permutation = torch.randperm(self.vocab_size, device=device, generator=generator)
        greenlist_size = self.greenlist_size()

        if self.config.select_green_tokens:
            return permutation[:greenlist_size]
        return permutation[self.vocab_size - greenlist_size :]

    def greenlist(self, prefix_tokens: Sequence[int]) -> list[int]:
        return self.greenlist_tensor(prefix_tokens, device=self.device).tolist()

    def is_green_token(self, token_id: int, prefix_tokens: Sequence[int]) -> bool:
        green_tokens = self.greenlist(prefix_tokens)
        return token_id in green_tokens

    def bias_logits(
        self, logits: Sequence[float], prefix_tokens: Sequence[int]
    ) -> list[float]:
        green_tokens = set(self.greenlist(prefix_tokens))
        return [
            logit + self.config.delta if token_id in green_tokens else logit
            for token_id, logit in enumerate(logits)
        ]

    def generate(
        self,
        prompt_tokens: Sequence[int],
        next_token_logits: TokenLogitsFn,
        max_new_tokens: int,
        rng: Random | None = None,
    ) -> list[int]:
        sampler = rng or Random()
        generated = list(prompt_tokens)

        for _ in range(max_new_tokens):
            logits = list(next_token_logits(generated))
            biased_logits = self.bias_logits(logits, generated)
            generated.append(sample_from_logits(biased_logits, sampler))

        return generated

    def _z_score(self, green_count: int, total_count: int) -> float:
        expected = self.config.gamma * total_count
        variance = total_count * self.config.gamma * (1.0 - self.config.gamma)
        return (green_count - expected) / sqrt(variance)

    def _score_standard(
        self,
        tokens: Sequence[int],
        *,
        return_green_token_mask: bool,
    ) -> tuple[int, int, list[bool] | None]:
        green_count = 0
        green_token_mask: list[bool] = []

        for index in range(1, len(tokens)):
            is_green = self.is_green_token(int(tokens[index]), tokens[:index])
            green_count += int(is_green)
            green_token_mask.append(is_green)

        if return_green_token_mask:
            return len(tokens) - 1, green_count, green_token_mask
        return len(tokens) - 1, green_count, None

    def _score_ignore_repeated_bigrams(self, tokens: Sequence[int]) -> tuple[int, int]:
        unique_bigrams = Counter(zip(tokens[:-1], tokens[1:]))
        green_count = 0

        for first_token, second_token in unique_bigrams:
            if self.is_green_token(int(second_token), [int(first_token)]):
                green_count += 1

        return len(unique_bigrams), green_count

    def score(
        self,
        tokens: Sequence[int],
        *,
        ignore_repeated_bigrams: bool = False,
        return_green_token_mask: bool = False,
    ) -> DetectionResult:
        if ignore_repeated_bigrams:
            num_tokens_scored, num_green_tokens = self._score_ignore_repeated_bigrams(
                tokens
            )
            green_token_mask = None
        else:
            num_tokens_scored, num_green_tokens, green_token_mask = self._score_standard(
                tokens,
                return_green_token_mask=return_green_token_mask,
            )

        z_score = self._z_score(num_green_tokens, num_tokens_scored)
        p_value = 1.0 - NormalDist().cdf(z_score)

        return DetectionResult(
            num_tokens_scored=num_tokens_scored,
            num_green_tokens=num_green_tokens,
            green_fraction=num_green_tokens / num_tokens_scored,
            z_score=z_score,
            p_value=p_value,
            prediction=z_score > self.config.z_threshold,
            green_token_mask=green_token_mask,
        )

    def detect(
        self,
        tokens: Sequence[int],
        *,
        z_threshold: float | None = None,
        ignore_repeated_bigrams: bool = False,
        return_green_token_mask: bool = False,
    ) -> DetectionResult:
        result = self.score(
            tokens,
            ignore_repeated_bigrams=ignore_repeated_bigrams,
            return_green_token_mask=return_green_token_mask,
        )
        threshold = self.config.z_threshold if z_threshold is None else z_threshold
        return DetectionResult(
            num_tokens_scored=result.num_tokens_scored,
            num_green_tokens=result.num_green_tokens,
            green_fraction=result.green_fraction,
            z_score=result.z_score,
            p_value=result.p_value,
            prediction=result.z_score > threshold,
            green_token_mask=result.green_token_mask,
        )
