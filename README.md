---
title: GestureFlow OpenEnv
emoji: ✋
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# GestureFlow OpenEnv

GestureFlow OpenEnv is a submission-ready OpenEnv-compatible environment for gesture-to-action workflow evaluation.  
It is designed for deployment on **Hugging Face Spaces (Docker SDK)** and provides typed APIs for:

- `reset()`
- `step()`
- `state()`

The project includes:

- 3 benchmark tasks: `easy`, `medium`, `hard`
- typed Pydantic request/response models
- grader-backed rewards and scores
- root-level `inference.py`
- Docker deployment support
- OpenEnv metadata through `openenv.yaml`

---

## Project Structure

```text
gesture-openenv/
├── Dockerfile
├── README.md
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
├── env/
│   ├── __init__.py
│   ├── environment.py
│   ├── grader.py
│   ├── models.py
│   └── tasks.py
└── server/
    ├── __init__.py
    └── app.py