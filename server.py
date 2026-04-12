"""
server.py
FastAPI HTTP server exposing the Clinical Trial Review OpenEnv.
Port 7860 is the HuggingFace Spaces default — we use it everywhere.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import ClinicalTrialReviewEnv
from env.models import Action, EnvState, Observation, StepResult
from env.protocols import PROTOCOL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("clinical-trial-env")

# ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clinical Trial Review — OpenEnv",
    description=(
        "OpenEnv environment for AI-driven clinical trial protocol review. "
        "Agents identify safety, regulatory, and statistical issues in "
        "synthetic but realistic trial documents."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# One shared environment instance
env = ClinicalTrialReviewEnv()


# ──────────────────────────────────────────────────────────────────
# Request schemas
# ──────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task_easy"


class StepRequest(BaseModel):
    action: Action


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "name": "Clinical Trial Review OpenEnv",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": "1.0.0", "environment": "clinical-trial-review"}


@app.post("/reset", response_model=Observation)
async def reset(request: ResetRequest) -> Observation:
    try:
        obs = env.reset(task_id=request.task_id)
        logger.info("reset task=%s", request.task_id)
        return obs
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("reset error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResult)
async def step(request: StepRequest) -> StepResult:
    try:
        result = env.step(request.action)
        logger.info(
            "step=%d reward=%.4f done=%s",
            result.info.get("step", 0),
            result.reward,
            result.done,
        )
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        logger.exception("step error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", response_model=EnvState)
async def state() -> EnvState:
    try:
        return env.state()
    except Exception as exc:
        logger.exception("state error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/tasks")
async def tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "id": tid,
                "metadata": data["metadata"],
                "max_steps": data["max_steps"],
                "num_sections": len(data["sections"]),
                "num_ground_truth_issues": len(data["ground_truth"]),
            }
            for tid, data in PROTOCOL_REGISTRY.items()
        ]
    }


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
