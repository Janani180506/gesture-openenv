from __future__ import annotations

from typing import Any, Dict, List


TASKS: Dict[str, List[Dict[str, Any]]] = {
    "easy": [
        {
            "id": "easy-1",
            "gestures": ["wave", "thumbs_up", "open_palm"],
            "targets": ["hello", "yes", "stop"],
            "context": {"mode": "literal", "domain": "assistive-basic"},
        },
        {
            "id": "easy-2",
            "gestures": ["namaste", "ok_sign", "clap"],
            "targets": ["greetings", "good", "great"],
            "context": {"mode": "literal", "domain": "assistive-basic"},
        },
        {
            "id": "easy-3",
            "gestures": ["pointing_up", "finger_gun", "clap"],
            "targets": ["one", "you", "great"],
            "context": {"mode": "literal", "domain": "assistive-basic"},
        },
    ],
    "medium": [
        {
            "id": "medium-1",
            "gestures": ["wave", "pointing_forward", "ok_sign", "clap", "open_palm"],
            "targets": ["hello", "go", "good", "great", "stop"],
            "context": {"mode": "sequence", "domain": "assistive-phrase"},
        },
        {
            "id": "medium-2",
            "gestures": ["thumbs_up", "cupped_hands", "ok_sign", "pointing_forward", "clap"],
            "targets": ["yes", "give", "good", "go", "great"],
            "context": {"mode": "sequence", "domain": "assistive-phrase"},
        },
        {
            "id": "medium-3",
            "gestures": ["fist", "shrug", "pointing_forward", "thumbs_down", "open_palm"],
            "targets": ["okay", "maybe", "go", "no", "stop"],
            "context": {"mode": "sequence", "domain": "assistive-phrase"},
        },
    ],
    "hard": [
        {
            "id": "hard-1",
            "gestures": ["wave", "open_palm", "wave", "finger_gun", "ok_sign", "clap", "open_palm"],
            "targets": ["hello", "stop", "hello", "you", "good", "great", "stop"],
            "context": {
                "mode": "ambiguous",
                "naive": ["hello", "hello", "hello", "you", "good", "great", "stop"],
                "explanation": "open_palm interrupts the greeting stream",
            },
        },
        {
            "id": "hard-2",
            "gestures": ["thumbs_up", "shrug", "thumbs_up", "crossed_arms", "pointing_forward", "thumbs_up", "open_palm"],
            "targets": ["yes", "maybe", "yes", "disagree", "go", "yes", "stop"],
            "context": {
                "mode": "ambiguous",
                "naive": ["yes", "yes", "yes", "yes", "go", "yes", "stop"],
                "explanation": "shrug and crossed_arms change certainty and agreement",
            },
        },
        {
            "id": "hard-3",
            "gestures": ["peace_sign", "shrug", "horns", "peace_sign", "wave", "thumbs_down", "clap"],
            "targets": ["peace", "maybe", "rock", "peace", "hello", "no", "great"],
            "context": {
                "mode": "ambiguous",
                "naive": ["peace", "peace", "rock", "peace", "hello", "great", "great"],
                "explanation": "middle gestures alter the natural literal pattern",
            },
        },
    ],
}


def get_task(task_name: str) -> List[Dict[str, Any]]:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task_name: {task_name}")
    return TASKS[task_name]


def task_ids() -> List[str]:
    return list(TASKS.keys())
