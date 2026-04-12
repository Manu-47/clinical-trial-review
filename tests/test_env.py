"""
tests/test_env.py
Full test suite — 28 tests covering all OpenEnv spec requirements.
Run: python -m pytest tests/test_env.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from env.environment import ClinicalTrialReviewEnv, VALID_TASKS
from env.models import Action, IssueReport
from env.protocols import PROTOCOL_REGISTRY, get_ground_truth
from graders.graders import (
    grade_task_easy, grade_task_medium, grade_task_hard,
    compute_step_reward, _grade_core,
)


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def env():
    return ClinicalTrialReviewEnv()


def _perfect_reports(task_id: str) -> list:
    gt = get_ground_truth(task_id)
    return [
        IssueReport(
            section_id=t["section_id"],
            issue_type=t["issue_type"],
            severity=t["severity"],
            description=t["description"],
        )
        for t in gt
    ]


def _easy_action(done: bool = False) -> Action:
    return Action(
        section_reviewed="IC_003",
        issues_reported=[
            IssueReport(
                section_id="IC_003",
                issue_type="missing_criteria",
                severity="critical",
                description=(
                    "No washout period specified for prior systemic anticancer therapy. "
                    "Standard practice requires at least 4 weeks or 5 half-lives."
                ),
            )
        ],
        done=done,
    )


# ══════════════════════════════════════════════════════════════════
# RESET TESTS
# ══════════════════════════════════════════════════════════════════

class TestReset:
    def test_easy_returns_observation(self, env):
        obs = env.reset("task_easy")
        assert obs.task_id == "task_easy"
        assert obs.step == 0
        assert len(obs.protocol_sections) == 5
        assert obs.issues_found_so_far == []

    def test_medium_returns_observation(self, env):
        obs = env.reset("task_medium")
        assert obs.task_id == "task_medium"
        assert len(obs.protocol_sections) == 5

    def test_hard_returns_observation(self, env):
        obs = env.reset("task_hard")
        assert obs.task_id == "task_hard"
        assert len(obs.protocol_sections) == 6

    def test_all_sections_pending_at_start(self, env):
        obs = env.reset("task_easy")
        section_ids = {s.section_id for s in obs.protocol_sections}
        assert set(obs.pending_sections) == section_ids

    def test_reset_clears_previous_state(self, env):
        env.reset("task_easy")
        env.step(_easy_action())
        obs = env.reset("task_easy")
        assert obs.step == 0
        assert obs.issues_found_so_far == []

    def test_invalid_task_raises_value_error(self, env):
        with pytest.raises(ValueError):
            env.reset("nonexistent_task")

    def test_all_valid_tasks_reset(self, env):
        for tid in VALID_TASKS:
            obs = env.reset(tid)
            assert obs.task_id == tid


# ══════════════════════════════════════════════════════════════════
# STEP TESTS
# ══════════════════════════════════════════════════════════════════

class TestStep:
    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.step(_easy_action())

    def test_step_returns_valid_result(self, env):
        env.reset("task_easy")
        result = env.step(_easy_action())
        assert 0.0 <= result.reward <= 1.0
        assert isinstance(result.done, bool)
        assert result.observation is not None
        assert result.info is not None

    def test_step_increments_counter(self, env):
        env.reset("task_easy")
        result = env.step(_easy_action())
        assert result.info["step"] == 1

    def test_done_when_agent_signals(self, env):
        env.reset("task_easy")
        result = env.step(Action(section_reviewed="IC_001", issues_reported=[], done=True))
        assert result.done is True

    def test_done_at_max_steps(self, env):
        env.reset("task_easy")  # max_steps=10
        result = None
        for _ in range(10):
            result = env.step(Action(section_reviewed="IC_001", issues_reported=[], done=False))
        assert result.done is True

    def test_step_after_done_raises(self, env):
        env.reset("task_easy")
        env.step(Action(section_reviewed="IC_001", issues_reported=[], done=True))
        with pytest.raises(RuntimeError):
            env.step(_easy_action())

    def test_reward_always_in_range(self, env):
        for tid in VALID_TASKS:
            env.reset(tid)
            for _ in range(3):
                result = env.step(Action(section_reviewed="X", issues_reported=[], done=False))
                assert 0.0 <= result.reward <= 1.0

    def test_echoed_message_reflects_action(self, env):
        env.reset("task_easy")
        result = env.step(_easy_action())
        assert "IC_003" in result.observation.echoed_message

    def test_pending_sections_decrease(self, env):
        obs = env.reset("task_easy")
        initial_pending = len(obs.pending_sections)
        result = env.step(Action(section_reviewed="IC_001", issues_reported=[], done=False))
        assert len(result.observation.pending_sections) <= initial_pending


# ══════════════════════════════════════════════════════════════════
# STATE TESTS
# ══════════════════════════════════════════════════════════════════

class TestState:
    def test_state_empty_before_reset(self, env):
        s = env.state()
        assert s.task_id == ""
        assert s.step == 0

    def test_state_after_reset(self, env):
        env.reset("task_medium")
        s = env.state()
        assert s.task_id == "task_medium"
        assert not s.episode_done

    def test_state_after_step(self, env):
        env.reset("task_easy")
        env.step(_easy_action())
        s = env.state()
        assert s.step == 1
        assert len(s.issues_found) == 1


# ══════════════════════════════════════════════════════════════════
# GRADER TESTS
# ══════════════════════════════════════════════════════════════════

class TestGraders:
    def test_empty_reports_score_low(self):
        gt = get_ground_truth("task_easy")
        score, r = grade_task_easy([], gt)
        assert 0.0 <= score <= 0.3
        assert 0.0 <= r.value <= 1.0

    def test_perfect_reports_score_high(self):
        gt = get_ground_truth("task_easy")
        score, r = grade_task_easy(_perfect_reports("task_easy"), gt)
        assert score >= 0.7

    def test_deterministic_easy(self):
        gt = get_ground_truth("task_easy")
        rep = _easy_action().issues_reported
        s1, _ = grade_task_easy(rep, gt)
        s2, _ = grade_task_easy(rep, gt)
        assert s1 == s2

    def test_deterministic_medium(self):
        gt = get_ground_truth("task_medium")
        reports = _perfect_reports("task_medium")
        s1, _ = grade_task_medium(reports, gt)
        s2, _ = grade_task_medium(reports, gt)
        assert s1 == s2

    def test_deterministic_hard(self):
        gt = get_ground_truth("task_hard")
        reports = _perfect_reports("task_hard")
        s1, _ = grade_task_hard(reports, gt)
        s2, _ = grade_task_hard(reports, gt)
        assert s1 == s2

    def test_score_always_in_01(self):
        for task_id, grader in [
            ("task_easy", grade_task_easy),
            ("task_medium", grade_task_medium),
            ("task_hard", grade_task_hard),
        ]:
            gt = get_ground_truth(task_id)
            # Adversarial: 20 false positives
            fp = [
                IssueReport(
                    section_id="X", issue_type="other", severity="critical",
                    description="completely fabricated nonexistent issue hallucination"
                )
            ] * 20
            score, r = grader(fp, gt)
            assert 0.0 <= score <= 1.0
            assert 0.0 <= r.value <= 1.0

    def test_false_positives_reduce_score(self):
        gt = get_ground_truth("task_easy")
        clean = [_easy_action().issues_reported[0]]
        s_clean, _ = grade_task_easy(clean, gt)
        with_fp = clean + [
            IssueReport(
                section_id="IC_001", issue_type="contradiction", severity="major",
                description="completely fabricated nonexistent hallucinated problem"
            )
        ]
        s_fp, _ = grade_task_easy(with_fp, gt)
        assert s_clean >= s_fp

    def test_perfect_all_tasks(self):
        for task_id, grader in [
            ("task_easy", grade_task_easy),
            ("task_medium", grade_task_medium),
            ("task_hard", grade_task_hard),
        ]:
            gt = get_ground_truth(task_id)
            score, _ = grader(_perfect_reports(task_id), gt)
            assert score >= 0.9, f"{task_id} perfect score too low: {score}"


# ══════════════════════════════════════════════════════════════════
# PROTOCOL REGISTRY TESTS
# ══════════════════════════════════════════════════════════════════

class TestProtocols:
    def test_all_tasks_registered(self):
        assert "task_easy" in PROTOCOL_REGISTRY
        assert "task_medium" in PROTOCOL_REGISTRY
        assert "task_hard" in PROTOCOL_REGISTRY

    def test_each_task_has_sections(self):
        for tid, data in PROTOCOL_REGISTRY.items():
            assert len(data["sections"]) > 0, f"{tid} has no sections"

    def test_each_task_has_ground_truth(self):
        for tid, data in PROTOCOL_REGISTRY.items():
            assert len(data["ground_truth"]) > 0, f"{tid} has no ground truth"

    def test_three_or_more_tasks(self):
        assert len(PROTOCOL_REGISTRY) >= 3

    def test_ground_truth_severities_valid(self):
        valid = {"critical", "major", "minor", "informational"}
        for tid, data in PROTOCOL_REGISTRY.items():
            for issue in data["ground_truth"]:
                assert issue["severity"] in valid, f"{tid}: bad severity {issue['severity']}"


# ══════════════════════════════════════════════════════════════════
# STEP REWARD TESTS
# ══════════════════════════════════════════════════════════════════

class TestStepReward:
    def test_new_section_bonus(self):
        gt = get_ground_truth("task_easy")
        r = compute_step_reward([], gt, [], "IC_001", 1, 10)
        assert r > 0

    def test_revisit_penalty(self):
        gt = get_ground_truth("task_easy")
        r = compute_step_reward([], gt, ["IC_001"], "IC_001", 2, 10)
        assert r < 0

    def test_correct_issue_positive(self):
        gt = get_ground_truth("task_easy")
        rep = _easy_action().issues_reported
        r = compute_step_reward(rep, gt, [], "IC_003", 1, 10)
        assert r > 0.1

    def test_false_positive_negative(self):
        gt = get_ground_truth("task_easy")
        fp = [IssueReport(
            section_id="IC_001", issue_type="other", severity="minor",
            description="completely fabricated nonexistent hallucinated problem made up"
        )]
        r = compute_step_reward(fp, gt, [], "IC_001", 1, 10)
        assert r < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
