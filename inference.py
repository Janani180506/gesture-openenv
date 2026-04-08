from __future__ import annotations

import asyncio
import json
import os
import textwrap
from statistics import mean
from typing import Any, Dict, List, Optional

from openai import OpenAI

from env.environment import GestureFlowEnv
from env.models import Action


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("GESTUREFLOW_BENCHMARK", "gestureflow_workflow")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.10"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "120"))
RUN_TASKS = [task.strip() for task in os.getenv("GESTUREFLOW_TASKS", "easy,medium,hard").split(",") if task.strip()]
EPISODES_PER_TASK = int(os.getenv("GESTUREFLOW_EPISODES", "2"))
TASK_SUCCESS_THRESHOLDS = {"easy": 0.60, "medium": 0.68, "hard": 0.72}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an agent operating a gesture-to-text RL environment.

    Return exactly one JSON object with the keys:
    predicted_text, confidence

    Rules:
    - Respect the current gesture, sequence history, and scenario context.
    - Predict ONLY the next token for the current step.
    - confidence must be a float between 0.0 and 1.0.
    - Never return markdown or explanations.
    """
).strip()


def divider(char: str = "=", width: int = 72) -> str:
    return char * width


def fmt_reward_list(values: List[float]) -> str:
    return ",".join(f"{v:.2f}" for v in values)


def log_start(task: str, env: str, model: str, episode: int) -> None:
    print(f"[START] task={task} env={env} model={model} episode={episode}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={fmt_reward_list(rewards)}",
        flush=True,
    )


def action_to_str(action: Action) -> str:
    return f"predict(predicted_text={action.predicted_text!r},confidence={action.confidence!r})"


def build_user_prompt(obs, task_name: str, step_num: int) -> str:
    return textwrap.dedent(
        f"""
        Task mode: {task_name}
        Step number: {step_num}
        Current gesture: {obs.gesture}
        Sequence so far: {json.dumps(obs.gesture_sequence, ensure_ascii=False)}
        Context history: {json.dumps(obs.context_history, ensure_ascii=False)}
        Expected sequence length: {obs.sequence_length}
        Scenario context: {json.dumps(obs.scenario_context, ensure_ascii=False)}
        Output JSON only.
        """
    ).strip()


def heuristic_action(obs, step_num: int) -> Action:
    mapping = {
        "thumbs_up": "yes", "thumbs_down": "no", "wave": "hello", "open_palm": "stop",
        "fist": "okay", "peace_sign": "peace", "pointing_up": "one", "pointing_forward": "go",
        "clap": "great", "shrug": "maybe", "ok_sign": "good", "pinch": "small",
        "spread_fingers": "five", "horns": "rock", "finger_gun": "you", "cupped_hands": "give",
        "raised_fist": "power", "namaste": "greetings", "hand_over_mouth": "quiet", "crossed_arms": "disagree",
    }
    return Action(predicted_text=mapping.get(obs.gesture, "unknown"), confidence=0.92)


def get_model_action(client: Optional[OpenAI], obs, task_name: str, step_num: int) -> Action:
    if client is None:
        return heuristic_action(obs, step_num)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(obs, task_name, step_num)},
            ],
        )
        raw = completion.choices[0].message.content.strip()
        data = json.loads(raw)
        return Action(
            predicted_text=str(data.get("predicted_text", "")).strip() or heuristic_action(obs, step_num).predicted_text,
            confidence=float(data.get("confidence", 0.85)),
        )
    except Exception:
        return heuristic_action(obs, step_num)


async def run_single_episode(client: Optional[OpenAI], task_name: str, episode: int) -> Dict[str, Any]:
    env = GestureFlowEnv(task_name=task_name)
    obs = env.reset()
    rewards: List[float] = []

    print(divider("-"), flush=True)
    print(f"TASK={task_name.upper()} | EPISODE={episode}/{EPISODES_PER_TASK}", flush=True)
    print(divider("-"), flush=True)

    log_start(task_name, BENCHMARK, MODEL_NAME, episode)

    for step in range(1, env.max_steps + 1):
        action = get_model_action(client, obs, task_name, step)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        log_step(step, action_to_str(action), reward, done, info.get("error"))
        if done:
            break

    score = env.grade()
    success = score >= TASK_SUCCESS_THRESHOLDS.get(task_name, 0.65)
    log_end(success, len(rewards), score, rewards)

    print(
        f"EPISODE_SUMMARY task={task_name} episode={episode} success={str(success).lower()} score={score:.2f} total_reward={sum(rewards):.2f}",
        flush=True,
    )

    return {
        "task": task_name,
        "episode": episode,
        "score": round(score, 4),
        "success": success,
        "steps": len(rewards),
        "rewards": [round(r, 4) for r in rewards],
        "total_reward": round(sum(rewards), 4),
    }


async def main() -> None:
    client = None
    if HF_TOKEN:
        try:
            client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        except Exception:
            client = None

    print(divider(), flush=True)
    print("GESTUREFLOW EVALUATION RUN", flush=True)
    print(divider(), flush=True)
    print(
        f"benchmark={BENCHMARK} model={MODEL_NAME} tasks={','.join(RUN_TASKS)} episodes_per_task={EPISODES_PER_TASK}",
        flush=True,
    )
    print(divider(), flush=True)

    all_episode_results: List[Dict[str, Any]] = []
    task_blocks: List[Dict[str, Any]] = []

    for task_name in RUN_TASKS:
        task_episode_results = []
        for episode in range(1, EPISODES_PER_TASK + 1):
            result = await run_single_episode(client, task_name, episode)
            task_episode_results.append(result)
            all_episode_results.append(result)

        avg_score = round(mean([item["score"] for item in task_episode_results]), 4)
        avg_steps = round(mean([item["steps"] for item in task_episode_results]), 2)
        success_count = sum(1 for item in task_episode_results if item["success"])
        task_summary = {
            "task": task_name,
            "episodes": task_episode_results,
            "average_score": avg_score,
            "average_steps": avg_steps,
            "success_count": success_count,
            "episodes_run": len(task_episode_results),
        }
        task_blocks.append(task_summary)

        print(divider("="), flush=True)
        print(
            f"TASK_SUMMARY task={task_name} episodes={len(task_episode_results)} success_count={success_count} average_score={avg_score:.2f} average_steps={avg_steps}",
            flush=True,
        )
        print(divider("="), flush=True)

    overall_average = round(mean([item["score"] for item in all_episode_results]), 4) if all_episode_results else 0.05
    overall_success = sum(1 for item in all_episode_results if item["success"])

    print(divider("#"), flush=True)
    print(
        f"FINAL_SUMMARY total_tasks={len(RUN_TASKS)} total_episodes={len(all_episode_results)} successful_episodes={overall_success} average_score={overall_average:.2f}",
        flush=True,
    )
    print(divider("#"), flush=True)

    print(
        json.dumps(
            {
                "benchmark": BENCHMARK,
                "model": MODEL_NAME,
                "episodes_per_task": EPISODES_PER_TASK,
                "tasks_run": task_blocks,
                "overall": {
                    "total_tasks": len(RUN_TASKS),
                    "total_episodes": len(all_episode_results),
                    "successful_episodes": overall_success,
                    "average_score": overall_average,
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
