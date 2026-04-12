"""
graders/graders.py
Deterministic graders for all three tasks.
Same input always produces the same score.
All scores are clamped to [0.0, 1.0].
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from env.models import IssueReport, Reward

SEVERITY_WEIGHTS: Dict[str, float] = {
    "critical": 3.0,
    "major": 2.0,
    "minor": 1.0,
    "informational": 0.5,
}

VALID_ISSUE_TYPES = {
    "missing_criteria", "contradiction", "safety_gap", "ethics_violation",
    "stats_flaw", "reporting_gap", "regulatory_noncompliance",
    "power_issue", "multiplicity_error", "other",
}

VALID_SEVERITIES = {"critical", "major", "minor", "informational"}


def _tokens(text: str) -> set:
    """Extract meaningful tokens (len > 4) from text for fuzzy matching."""
    cleaned = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return {t for t in cleaned.split() if len(t) > 4}


def _is_match(reported: IssueReport, truth: Dict[str, Any]) -> bool:
    """
    True if reported issue matches a ground-truth issue.
    Rules:
      1. section_id must match exactly
      2. issue_type must match exactly
      3. At least 2 meaningful tokens overlap between descriptions
    """
    if reported.section_id.strip() != truth["section_id"].strip():
        return False
    if reported.issue_type.strip() != truth["issue_type"].strip():
        return False
    truth_tokens = _tokens(truth["description"])
    report_tokens = _tokens(reported.description)
    return len(truth_tokens & report_tokens) >= 2


def _grade_core(
    reported: List[IssueReport],
    ground_truth: List[Dict[str, Any]],
) -> Tuple[float, Reward]:
    """
    Core grading engine shared by all three task graders.
    Returns (score in [0,1], Reward object).
    """
    if not ground_truth:
        return 0.0, Reward(value=0.0, feedback="No ground truth defined.")

    max_possible = sum(SEVERITY_WEIGHTS.get(t["severity"], 1.0) for t in ground_truth)
    if max_possible <= 0:
        max_possible = 1.0

    matched: set = set()
    raw = 0.0
    fp_count = 0
    correct = 0
    breakdown: Dict[str, float] = {}

    for rep in reported:
        if rep.issue_type not in VALID_ISSUE_TYPES:
            fp_count += 1
            raw -= 0.3
            continue
        if rep.severity not in VALID_SEVERITIES:
            fp_count += 1
            raw -= 0.3
            continue

        hit = False
        for idx, truth in enumerate(ground_truth):
            if idx in matched:
                continue
            if _is_match(rep, truth):
                w = SEVERITY_WEIGHTS.get(truth["severity"], 1.0)
                raw += w
                matched.add(idx)
                correct += 1
                cat = truth["issue_type"]
                breakdown[cat] = breakdown.get(cat, 0.0) + w
                hit = True
                break

        if not hit:
            fp_count += 1
            raw -= 0.5

    missed_crit = 0
    for idx, truth in enumerate(ground_truth):
        if idx not in matched and truth["severity"] == "critical":
            raw -= 2.0
            missed_crit += 1

    raw = max(raw, 0.0)
    score = min(raw / max_possible, 1.0)
    score = max(score, 0.0)

    bk_norm = {k: round(min(v / max_possible, 1.0), 4) for k, v in breakdown.items()}
    total = len(ground_truth)
    missed = total - len(matched)

    parts = [
        f"Found {correct}/{total} real issues.",
        f"False positives: {fp_count}.",
        f"Missed critical: {missed_crit}.",
    ]
    if score >= 0.85:
        parts.append("Excellent review!")
    elif score >= 0.60:
        parts.append("Good review — some gaps remain.")
    elif score >= 0.30:
        parts.append("Partial review — significant gaps.")
    else:
        parts.append("Poor review — most issues missed.")

    return round(score, 4), Reward(
        value=round(score, 4),
        breakdown=bk_norm,
        feedback=" ".join(parts),
        correct_issues_found=correct,
        false_positives=fp_count,
        missed_critical=missed_crit,
    )


def grade_task_easy(
    reported: List[IssueReport],
    ground_truth: List[Dict[str, Any]],
) -> Tuple[float, Reward]:
    return _grade_core(reported, ground_truth)


def grade_task_medium(
    reported: List[IssueReport],
    ground_truth: List[Dict[str, Any]],
) -> Tuple[float, Reward]:
    score, reward = _grade_core(reported, ground_truth)
    # Extra penalty for missing regulatory violations (highest stakes)
    reg_misses = sum(
        1 for idx, t in enumerate(ground_truth)
        if t["issue_type"] == "regulatory_noncompliance"
        and not any(_is_match(r, t) for r in reported)
    )
    if reg_misses:
        score = max(0.0, round(score - 0.05 * reg_misses, 4))
        reward.value = score
        reward.feedback += f" Regulatory penalty: {reg_misses} violation(s) missed."
    return score, reward


def grade_task_hard(
    reported: List[IssueReport],
    ground_truth: List[Dict[str, Any]],
) -> Tuple[float, Reward]:
    score, reward = _grade_core(reported, ground_truth)
    # Bonus for explicitly quantifying FWER
    for r in reported:
        dl = r.description.lower()
        if any(kw in dl for kw in ("fwer", "familywise", "family-wise", "33.7", "33.6")):
            score = min(1.0, round(score + 0.05, 4))
            reward.value = score
            reward.feedback += " Bonus: FWER quantification identified."
            break
    return score, reward


def compute_step_reward(
    reported: List[IssueReport],
    ground_truth: List[Dict[str, Any]],
    reviewed_so_far: List[str],
    section_reviewed: str,
    step: int,
    max_steps: int,
) -> float:
    """
    Per-step partial reward (gives learning signal every step, not just at end).
    Returns a float — NOT clamped here; caller adds to cumulative.
    """
    sr = 0.0

    if section_reviewed not in reviewed_so_far:
        sr += 0.05   # exploration bonus
    else:
        sr -= 0.02   # redundancy penalty

    for rep in reported:
        matched = any(_is_match(rep, t) for t in ground_truth)
        if matched:
            for t in ground_truth:
                if _is_match(rep, t):
                    sev = t.get("severity", "minor")
                    sr += {"critical": 0.25, "major": 0.18, "minor": 0.12}.get(sev, 0.10)
                    break
        else:
            sr -= 0.10

    return round(max(sr, -0.5), 4)
