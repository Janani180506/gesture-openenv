from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException

from env.environment import GestureFlowEnv
from env.models import Action


ENV = GestureFlowEnv(task_name="hard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="GestureFlow",
    version="2.1.0",
    description="OpenEnv-compatible gesture-to-text workflow environment.",
    lifespan=lifespan,
)


@app.get("/")
def root() -> Dict[str, object]:
    return {
        "name": "GestureFlow",
        "status": "ok",
        "version": "2.1.0",
        "entrypoint": "server.app:app",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "healthy",
        "task_name": ENV.task_name,
        "done": ENV.done,
        "max_steps": ENV.max_steps,
    }


@app.get("/tasks")
def tasks() -> Dict[str, object]:
    return {"tasks": ENV.task_catalog()}


@app.post("/reset")
def reset(task_name: str = "hard") -> Dict[str, object]:
    try:
        observation = ENV.reset(task_name=task_name)
        return {
            "observation": observation.model_dump(),
            "task_name": ENV.task_name,
            "episode_id": ENV.episode_id,
            "max_steps": ENV.max_steps,
            "success_threshold": ENV.success_threshold,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(action: Action) -> Dict[str, object]:
    try:
        obs, reward, done, info = ENV.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": float(f"{reward:.4f}"),
            "done": done,
            "info": info,
            "score": float(f"{ENV.grade():.4f}"),
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state() -> Dict[str, object]:
    return ENV.state()


def main() -> None:
    import os
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
