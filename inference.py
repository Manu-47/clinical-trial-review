"""
inference.py
Baseline inference script for Clinical Trial Review OpenEnv.

Reads configuration from environment variables:
  API_BASE_URL   - LLM API endpoint
  MODEL_NAME     - model identifier
  HF_TOKEN       - API key (also checked as OPENAI_API_KEY)
  ENV_BASE_URL   - running environment server URL

Emits structured JSON logs:
  {"type": "START", ...}
  {"type": "STEP",  ...}
  {"type": "END",   ...}

Usage:
  export API_BASE_URL="https://api.openai.com/v1"
  export MODEL_NAME="gpt-4o"
  export HF_TOKEN="sk-..."
  export ENV_BASE_URL="http://localhost:7860"
  python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
API_KEY: str = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)
ENV_BASE_URL: str = os.environ.get(
    "ENV_BASE_URL", "http://localhost:7860"
).rstrip("/")

TEMPERATURE: float = 0.0
MAX_TOKENS: int = 1000
SUCCESS_THRESHOLD: float = 0.5
BENCHMARK: str = "clinical-trial-review"

TASKS: List[str] = ["task_easy", "task_medium", "task_hard"]
MAX_STEPS: Dict[str, int] = {
    "task_easy": 10,
    "task_medium": 15,
    "task_hard": 20,
}

HTTP_TIMEOUT: float = 60.0   # seconds per request
SERVER_WAIT: float = 5.0     # seconds to wait between health retries
SERVER_RETRIES: int = 12     # health check retries (12 × 5s = 60s max)

# ──────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert clinical trial protocol reviewer with 20+ years of experience "
    "in FDA regulatory affairs, clinical research, and biostatistics. "
    "Your job is to review sections of a clinical trial protocol and identify ALL issues.\n\n"
    "Respond with a JSON object ONLY — no markdown fences, no explanation, just raw JSON.\n\n"
    "Required JSON format:\n"
    "{\n"
    '  "section_reviewed": "<section_id>",\n'
    '  "issues_reported": [\n'
    '    {\n'
    '      "section_id": "<section_id>",\n'
    '      "issue_type": "<missing_criteria|contradiction|safety_gap|ethics_violation|'
    'stats_flaw|reporting_gap|regulatory_noncompliance|power_issue|multiplicity_error|other>",\n'
    '      "severity": "<critical|major|minor|informational>",\n'
    '      "description": "<specific description with medical/regulatory terminology>",\n'
    '      "recommendation": "<specific fix>",\n'
    '      "regulation_ref": "<e.g. 21 CFR 312.32 or ICH E9>"\n'
    '    }\n'
    "  ],\n"
    '  "next_section_requested": "<section_id or null>",\n'
    '  "reasoning": "<brief reasoning>",\n'
    '  "done": <true|false>\n'
    "}\n\n"
    "Be thorough. Identify:\n"
    "- Missing eligibility criteria (washout periods, organ function thresholds, QTc)\n"
    "- Safety monitoring gaps (AE windows, DSMB notification timelines)\n"
    "- Statistical flaws (multiplicity, power, missing data assumptions, FWER)\n"
    "- Regulatory non-compliance (21 CFR, ICH guidelines)\n"
    "Do NOT fabricate issues. Only report genuine protocol deficiencies."
)

# ──────────────────────────────────────────────────────────────────
# Structured logging — exact field names required by evaluator
# ──────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(
        json.dumps({
            "type": "START",
            "task": task,
            "env": env,
            "model": model,
            "timestamp": time.time(),
        }),
        flush=True,
    )


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    print(
        json.dumps({
            "type": "STEP",
            "step": step,
            "action": action,
            "reward": reward,
            "done": done,
            "error": error,
            "timestamp": time.time(),
        }),
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    print(
        json.dumps({
            "type": "END",
            "success": success,
            "steps": steps,
            "score": score,
            "rewards": rewards,
            "timestamp": time.time(),
        }),
        flush=True,
    )


# ──────────────────────────────────────────────────────────────────
# LLM call — fully wrapped
# ──────────────────────────────────────────────────────────────────

def get_model_action(
    client: OpenAI,
    step: int,
    observation: Dict[str, Any],
    last_reward: float,
    history: List[str],
) -> str:
    """Call LLM and return raw action JSON string. Never raises."""
    try:
        sections_text = "\n\n".join(
            f"=== {s.get('section_id', '?')}: {s.get('title', '?')} ===\n{s.get('content', '')}"
            for s in observation.get("protocol_sections", [])
        )

        pending = observation.get("pending_sections", [])
        found = observation.get("issues_found_so_far", [])
        task_id = observation.get("task_id", "unknown")

        user_prompt = (
            f"STEP {step} — Task: {task_id}\n"
            f"Environment message: {observation.get('message', '')}\n"
            f"Last reward: {last_reward:+.4f}\n\n"
            f"Pending sections (not yet reviewed): "
            f"{', '.join(pending) if pending else 'None — all reviewed'}\n\n"
            f"Issues reported so far:\n"
            + ("\n".join(found) if found else "None yet.")
            + f"\n\nRecent history:\n"
            + ("\n".join(history[-3:]) if history else "First step.")
            + f"\n\nPROTOCOL SECTIONS:\n{sections_text}\n\n"
            "Review the next pending section. "
            "If all sections are reviewed and you are confident, set \"done\": true. "
            "Respond ONLY with valid JSON."
        )

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip accidental markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()

        if not text:
            raise ValueError("Empty response from model")

        # Validate it's parseable JSON
        json.loads(text)
        return text

    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        # Return a valid no-op action so the episode continues
        fallback_section = (
            observation.get("pending_sections", ["UNKNOWN"])[0]
            if observation.get("pending_sections")
            else "UNKNOWN"
        )
        return json.dumps({
            "section_reviewed": fallback_section,
            "issues_reported": [],
            "next_section_requested": None,
            "reasoning": f"Fallback due to error: {exc}",
            "done": False,
        })


# ──────────────────────────────────────────────────────────────────
# Environment HTTP client — fully wrapped
# ──────────────────────────────────────────────────────────────────

def env_reset(http: httpx.Client, task_id: str) -> Dict[str, Any]:
    """Call /reset. Raises on HTTP error."""
    resp = http.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(http: httpx.Client, action_json: str) -> Dict[str, Any]:
    """Call /step. Returns a valid StepResult dict or raises."""
    try:
        action_dict = json.loads(action_json)
    except json.JSONDecodeError as exc:
        print(f"[DEBUG] Bad JSON from model — using no-op: {exc}", flush=True)
        action_dict = {
            "section_reviewed": "UNKNOWN",
            "issues_reported": [],
            "next_section_requested": None,
            "reasoning": "JSON parse error",
            "done": True,
        }

    resp = http.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action_dict},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def wait_for_server(http: httpx.Client) -> bool:
    """Retry /health until the server is up. Returns True on success."""
    for attempt in range(1, SERVER_RETRIES + 1):
        try:
            resp = http.get(f"{ENV_BASE_URL}/health", timeout=10.0)
            if resp.status_code == 200:
                print(
                    f"[INFO] Environment ready: {resp.json()}",
                    flush=True,
                )
                return True
        except Exception as exc:
            print(
                f"[INFO] Health check attempt {attempt}/{SERVER_RETRIES}: {exc}",
                flush=True,
            )
        time.sleep(SERVER_WAIT)
    return False


# ──────────────────────────────────────────────────────────────────
# Run one task episode
# ──────────────────────────────────────────────────────────────────

def run_task(task_id: str, client: OpenAI, http: httpx.Client) -> float:
    """
    Run one full episode for task_id.
    Returns final score in [0, 1].
    Guaranteed not to raise — catches all exceptions internally.
    """
    max_steps = MAX_STEPS[task_id]
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    observation: Dict[str, Any] = {}

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Reset ─────────────────────────────────────────────────
        try:
            observation = env_reset(http, task_id)
        except Exception as exc:
            print(f"[DEBUG] reset() failed for {task_id}: {exc}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return 0.0

        last_reward = 0.0
        done = False

        # ── Episode loop ──────────────────────────────────────────
        for step in range(1, max_steps + 1):
            if done:
                break

            # Get action from LLM
            action_json = get_model_action(
                client, step, observation, last_reward, history
            )

            # Step environment
            error_msg = None
            try:
                result = env_step(http, action_json)
            except Exception as exc:
                error_msg = str(exc)
                print(f"[DEBUG] step() failed at step {step}: {exc}", flush=True)
                # Try to continue with a done=True fallback
                try:
                    done_action = json.dumps({
                        "section_reviewed": "UNKNOWN",
                        "issues_reported": [],
                        "done": True,
                    })
                    result = env_step(http, done_action)
                except Exception:
                    log_step(step=step, action=action_json[:200], reward=0.0, done=True, error=error_msg)
                    break

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            observation = result.get("observation", observation)
            last_reward = reward

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_json[:300],
                reward=reward,
                done=done,
                error=error_msg,
            )

            history.append(f"Step {step}: reward {reward:+.4f}")

        # ── Final score ───────────────────────────────────────────
        # The last reward when done=True is the graded final score
        if rewards:
            score = rewards[-1] if done else rewards[-1]
        score = float(max(0.0, min(1.0, score)))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(
            f"[DEBUG] Unhandled exception in run_task({task_id}): {exc}",
            flush=True,
        )
        traceback.print_exc()
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Validate configuration ────────────────────────────────────
    if not API_KEY:
        print(
            "[ERROR] No API key found. "
            "Set HF_TOKEN or OPENAI_API_KEY environment variable.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    print(f"[INFO] API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"[INFO] MODEL_NAME   : {MODEL_NAME}", flush=True)
    print(f"[INFO] ENV_BASE_URL : {ENV_BASE_URL}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    with httpx.Client(timeout=HTTP_TIMEOUT) as http:

        # ── Wait for environment server ───────────────────────────
        if not wait_for_server(http):
            print(
                "[ERROR] Environment server not reachable after "
                f"{SERVER_RETRIES * SERVER_WAIT:.0f}s. "
                f"Is it running at {ENV_BASE_URL}?",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)

        # ── Run all three tasks ───────────────────────────────────
        all_scores: Dict[str, float] = {}

        for task_id in TASKS:
            print(f"\n{'=' * 60}", flush=True)
            print(f"[INFO] Running task: {task_id}", flush=True)
            print(f"{'=' * 60}", flush=True)

            try:
                score = run_task(task_id, client, http)
            except Exception as exc:
                print(f"[DEBUG] run_task raised: {exc}", flush=True)
                score = 0.0

            all_scores[task_id] = score
            print(f"[INFO] {task_id} score: {score:.4f}", flush=True)

        # ── Summary ───────────────────────────────────────────────
        print(f"\n{'=' * 60}", flush=True)
        print("[SUMMARY] Baseline Results", flush=True)
        for tid, sc in all_scores.items():
            status = "PASS" if sc >= SUCCESS_THRESHOLD else "FAIL"
            print(f"  [{status}] {tid}: {sc:.4f}", flush=True)
        avg = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
        print(f"  Average: {avg:.4f}", flush=True)
        print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[FATAL] Unhandled top-level exception: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
