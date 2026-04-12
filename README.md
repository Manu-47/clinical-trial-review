---
title: Clinical Trial Review OpenEnv
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - rl
  - agent
  - healthcare
  - nlp
  - regulatory
  - safety
---

# 🏥 Clinical Trial Review — OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://openenv.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)

> **An OpenEnv environment where AI agents review clinical trial protocols and identify safety gaps, regulatory violations, and statistical flaws.**

---

## Why This Exists

Every clinical trial protocol must be reviewed before drugs are tested on humans. FDA reviewers, IRB committees, and pharmaceutical QA teams spend thousands of hours hunting for:

- **Safety gaps** — missing monitoring procedures that let harms go unreported
- **Statistical flaws** — underpowered trials, uncontrolled FWER, bad missing-data assumptions
- **Regulatory violations** — non-compliance with 21 CFR, ICH guidelines
- **Ethics issues** — inadequate contraception requirements, missing informed consent protections

This environment teaches AI agents to do this review. A capable agent here could assist real FDA reviewers and IRB committees — reducing review time and catching issues that fatigued humans miss.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/reset` | Start episode `{"task_id": "task_easy"}` |
| POST | `/step` | Take action `{"action": {...}}` |
| GET | `/state` | Full internal state |
| GET | `/tasks` | List all tasks |
| GET | `/docs` | Swagger UI |

---

## Three Tasks

### Task 1 — Inclusion/Exclusion Criteria (Easy)
- **Protocol**: Phase I oncology trial (XRT-447)
- **Sections**: 5 | **Max steps**: 10 | **Issues**: 5 (2 critical)
- Find missing washout period, QTc threshold, hepatic impairment criteria, contraception duration, vague organ function definition

### Task 2 — AE Reporting & DSMB Gaps (Medium)
- **Protocol**: Phase II rheumatology trial (MRL-892), 12 sites
- **Sections**: 5 | **Max steps**: 15 | **Issues**: 6 (3 critical)
- Find short AE collection window, DSMB 21 CFR violation, missing opportunistic infection stopping rule, undefined meeting triggers, no central safety database

### Task 3 — Statistical Analysis Plan Audit (Hard)
- **Protocol**: Phase III diabetes registration trial (ZNT-101)
- **Sections**: 6 | **Max steps**: 20 | **Issues**: 8 (3 critical)
- Find MNAR assumption flaw, uncontrolled FWER (33.7%), underpowered sample size, low power, optimistic dropout, contradictory alpha spending, lenient futility, subgroup multiplicity

---

## Quickstart

### Docker
```bash
docker build -t clinical-trial-env .
docker run -p 7860:7860 clinical-trial-env
```

### Local Python
```bash
pip install -r requirements.txt
python server.py
```

### Run Inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your-api-key"
export ENV_BASE_URL="http://localhost:7860"
python inference.py
```

---

## Baseline Scores (GPT-4o, temperature=0)

| Task | Score |
|------|-------|
| task_easy | ~0.72 |
| task_medium | ~0.58 |
| task_hard | ~0.41 |
| **Average** | **~0.57** |

---

## OpenEnv Spec Compliance

- ✅ Typed Pydantic models (Observation, Action, Reward, StepResult, EnvState)
- ✅ `reset()` → Observation
- ✅ `step()` → (observation, reward, done, info)
- ✅ `state()` → EnvState
- ✅ `openenv.yaml` with full metadata
- ✅ Rewards clamped to [0.0, 1.0]
- ✅ 3+ tasks with programmatic, deterministic graders
- ✅ Partial reward every step + final grade at episode end
- ✅ Dockerfile builds and runs on port 7860
- ✅ Baseline `inference.py` with START/STEP/END logging
