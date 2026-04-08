from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, Dict, List, Tuple
from uuid import uuid4

from env.grader import compute_reward, final_grade
from env.models import Action, EnvState, Observation
from env.tasks import TASKS, get_task


SUCCESS_THRESHOLDS = {"easy": 0.60, "medium": 0.68, "hard": 0.72}


class GestureFlowEnv:
    _scenario_cursors: ClassVar[Dict[str, int]] = {"easy": 0, "medium": 0, "hard": 0}

    def __init__(self, task_name: str = "hard") -> None:
        self.task_name = task_name
        self.max_steps = 0
        self.success_threshold = SUCCESS_THRESHOLDS.get(task_name, 0.65)
        self.scenario: Dict[str, Any] = {}
        self.step_index = 0
        self.predictions: List[str] = []
        self.cumulative_reward = 0.0
        self.done = False
        self.episode_id = ""
        self.reset(task_name=task_name)

    def task_catalog(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": task_id,
                "description": description,
                "max_steps": max_steps,
                "difficulty": task_id,
                "success_threshold": SUCCESS_THRESHOLDS[task_id],
                "scenario_count": len(TASKS[task_id]),
            }
            for task_id, description, max_steps in [
                ("easy", "Short literal gesture sequences", 3),
                ("medium", "Longer gesture phrase recognition", 5),
                ("hard", "Ambiguous context-sensitive gesture resolution", 7),
            ]
        ]

    def reset(self, task_name: str | None = None) -> Observation:
        if task_name is not None:
            if task_name not in SUCCESS_THRESHOLDS:
                raise ValueError(f"Unknown task_name: {task_name}")
            self.task_name = task_name

        scenarios = get_task(self.task_name)
        cursor = self._scenario_cursors[self.task_name] % len(scenarios)
        self.scenario = deepcopy(scenarios[cursor])
        self._scenario_cursors[self.task_name] = (cursor + 1) % len(scenarios)

        self.max_steps = len(self.scenario["gestures"])
        self.success_threshold = SUCCESS_THRESHOLDS.get(self.task_name, 0.65)
        self.step_index = 0
        self.predictions = []
        self.cumulative_reward = 0.0
        self.done = False
        self.episode_id = str(uuid4())[:8]
        return self._observation()

    def _observation(self) -> Observation:
        current_index = min(self.step_index, self.max_steps - 1)
        current_gesture = self.scenario["gestures"][current_index]
        return Observation(
            gesture=current_gesture,
            gesture_sequence=self.scenario["gestures"][: current_index + 1],
            context_history=list(self.predictions),
            step_number=self.step_index,
            task_name=self.task_name,
            sequence_length=self.max_steps,
            scenario_context=deepcopy(self.scenario.get("context", {})),
            available_actions=["predict"],
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode already completed. Call reset() first.")

        target = self.scenario["targets"][self.step_index]
        reward = compute_reward(
            predicted_text=action.predicted_text,
            target_text=target,
            task_name=self.task_name,
            step_index=self.step_index,
            scenario_context=self.scenario.get("context", {}),
        )
        self.predictions.append(action.predicted_text)
        self.cumulative_reward += reward
        self.step_index += 1
        self.done = self.step_index >= self.max_steps

        info = {
            "episode_id": self.episode_id,
            "scenario_id": self.scenario.get("id", "unknown"),
            "step": self.step_index,
            "target": target,
            "cumulative_reward": round(self.cumulative_reward, 4),
            "task_name": self.task_name,
            "error": None,
        }
        return self._observation(), reward, self.done, info

    def grade(self) -> float:
        return final_grade(
            predictions=self.predictions,
            targets=self.scenario["targets"],
            task_name=self.task_name,
            scenario_context=self.scenario.get("context", {}),
        )

    def state(self) -> Dict[str, Any]:
        current_index = min(self.step_index, self.max_steps - 1)
        current_gesture = self.scenario["gestures"][current_index]
        return EnvState(
            task_name=self.task_name,
            step=self.step_index,
            max_steps=self.max_steps,
            current_gesture=current_gesture,
            gesture_sequence=list(self.scenario["gestures"][: current_index + 1]),
            context_history=list(self.predictions),
            target_sequence=list(self.scenario["targets"]),
            cumulative_reward=round(self.cumulative_reward, 4),
            done=self.done,
            episode_id=self.episode_id,
            success_threshold=self.success_threshold,
        ).model_dump()
