from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Optional


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _similarity(a: Optional[str], b: Optional[str]) -> float:
    a_norm, b_norm = _norm(a), _norm(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _strict_open_interval(score: float, low: float = 0.05, high: float = 0.95) -> float:
    score = max(0.0, min(1.0, score))
    return round(low + (high - low) * score, 4)


def compute_reward(predicted_text: str, target_text: str, task_name: str, step_index: int, scenario_context: Optional[Dict] = None) -> float:
    sim = _similarity(predicted_text, target_text)
    base = 0.0
    if sim >= 0.995:
        base = 0.96
    elif sim >= 0.80:
        base = 0.72 + 0.18 * sim
    elif sim >= 0.55:
        base = 0.40 + 0.28 * sim
    elif sim > 0.0:
        base = 0.12 + 0.20 * sim
    else:
        base = 0.03

    if task_name == "hard" and scenario_context:
        naive = scenario_context.get("naive", [])
        naive_token = naive[step_index] if step_index < len(naive) else None
        if _norm(predicted_text) == _norm(target_text) and _norm(predicted_text) != _norm(naive_token):
            base += 0.02
        elif _norm(predicted_text) == _norm(naive_token) and _norm(predicted_text) != _norm(target_text):
            base -= 0.05

    base = max(0.01, min(0.99, base))
    return round(base, 4)


def final_grade(predictions: List[str], targets: List[str], task_name: str, scenario_context: Optional[Dict] = None) -> float:
    if not predictions or not targets:
        return 0.05

    rewards = [
        compute_reward(predictions[i], targets[i], task_name, i, scenario_context)
        for i in range(min(len(predictions), len(targets)))
    ]
    mean_reward = sum(rewards) / len(rewards)
    return _strict_open_interval(mean_reward)
