from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    gesture: str = Field(..., description="Current gesture label")
    gesture_sequence: List[str] = Field(default_factory=list, description="Sequence of gestures seen so far")
    context_history: List[str] = Field(default_factory=list, description="Predictions produced in previous steps")
    step_number: int = Field(default=0, description="Zero-based step index")
    task_name: str = Field(default="easy", description="Task identifier: easy | medium | hard")
    sequence_length: int = Field(default=1, description="Total length of the current gesture sequence")
    scenario_context: Dict[str, Any] = Field(default_factory=dict, description="Extra context for ambiguous tasks")
    available_actions: List[str] = Field(default_factory=lambda: ["predict"], description="Allowed actions")


class Action(BaseModel):
    predicted_text: str = Field(..., description="Predicted token for the current gesture")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Agent confidence")


class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=0.99, description="Strictly bounded reward in (0,1)")
    reason: str = Field(..., description="Explanation of the reward")
    partial_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_correct: bool = Field(default=False)
    is_partial: bool = Field(default=False)


class EnvState(BaseModel):
    task_name: str
    step: int
    max_steps: int
    current_gesture: str
    gesture_sequence: List[str]
    context_history: List[str]
    target_sequence: List[str]
    cumulative_reward: float
    done: bool
    episode_id: str
    success_threshold: float
