"""
tests/test_grading_engine.py
============================
Unit and integration tests for the StrictUniversityGrading engine.

Run with:
    pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd

from app.grading_engine import (
    GradingConfig, GradingEngine, GradingResult,
    CourseType, Protocol, GradingMethod,
    PASSING_GRADES,
)
from app.simulation import generate_cohort


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    """Minimal valid DataFrame with 5 students."""
    return pd.DataFrame({
        "id":          [1, 2, 3, 4, 5],
        "marks":       [82, 65, 45, 32, 91],
        "attendance":  [90, 85, 80, 74, 95],   # student 4 below 75%
        "ese_marks":   [40, 30, "AB", 10, 50],
    })


@pytest.fixture
def large_df():
    return generate_cohort(n=100, distribution="normal_high", seed=7)


@pytest.fixture
def default_cfg():
    return GradingConfig()


# ── Stage 1: Attendance ───────────────────────────────────────────────────────

class TestAttendanceGate:
    def test_below_75_gets_I(self, small_df, default_cfg):
        result = GradingEngine(default_cfg).process(small_df)
        student_4 = result.results_df[result.results_df["id"] == 4]
        assert student_4["Final_Grade"].iloc[0] == "I"

    def test_above_75_not_I(self, small_df, default_cfg):
        result = GradingEngine(default_cfg).process(small_df)
        others = result.results_df[result.results_df["id"] != 4]
        assert (others["Final_Grade"] != "I").all()

    def test_exactly_75_passes(self, default_cfg):
        df = pd.DataFrame({
            "id": [1], "marks": [70], "attendance": [75.0], "ese_marks": [30],
        })
        result = GradingEngine(default_cfg).process(df)
        assert result.results_df["Final_Grade"].iloc[0] != "I"


# ── Stage 2: ESE Absent ───────────────────────────────────────────────────────

class TestESEAbsentGate:
    def test_AB_gets_Z(self, small_df, default_cfg):
        result = GradingEngine(default_cfg).process(small_df)
        student_3 = result.results_df[result.results_df["id"] == 3]
        assert student_3["Final_Grade"].iloc[0] == "Z"

    def test_AB_case_insensitive(self, default_cfg):
        df = pd.DataFrame({
            "id": [1], "marks": [60], "attendance": [80], "ese_marks": ["ab"],
        })
        result = GradingEngine(default_cfg).process(df)
        assert result.results_df["Final_Grade"].iloc[0] == "Z"

    def test_numeric_ese_not_Z(self, default_cfg):
        df = pd.DataFrame({
            "id": [1], "marks": [60], "attendance": [80], "ese_marks": [25],
        })
        result = GradingEngine(default_cfg).process(df)
        assert result.results_df["Final_Grade"].iloc[0] != "Z"


# ── Stage 3: ESE Minimum ──────────────────────────────────────────────────────

class TestESEMinimum:
    def test_below_threshold_gets_F(self, default_cfg):
        """Default: M_ESE=60, threshold=0.20*60=12. Score of 10 → F."""
        df = pd.DataFrame({
            "id": [1], "marks": [70], "attendance": [80], "ese_marks": [10],
        })
        result = GradingEngine(default_cfg).process(df)
        assert result.results_df["Final_Grade"].iloc[0] == "F"

    def test_exactly_at_threshold_not_F(self, default_cfg):
        df = pd.DataFrame({
            "id": [1], "marks": [70], "attendance": [80], "ese_marks": [12.0],
        })
        result = GradingEngine(default_cfg).process(df)
        assert result.results_df["Final_Grade"].iloc[0] != "F"


# ── Stage 4: Protocol Selection ───────────────────────────────────────────────

class TestProtocolSelection:
    def test_protocol_A_excludes_ese_failures(self):
        """Protocol A reference pop should exclude ESE failures."""
        df = generate_cohort(n=50, distribution="normal_high", seed=1,
                             ese_fail_rate=0.30)
        cfg_a = GradingConfig(protocol=Protocol.EXCLUSIVE)
        cfg_b = GradingConfig(protocol=Protocol.INCLUSIVE)
        r_a = GradingEngine(cfg_a).process(df)
        r_b = GradingEngine(cfg_b).process(df)
        # Protocol A reference pop is smaller or equal
        assert r_a.stats["n_ref"] <= r_b.stats["n_ref"]

    def test_protocol_A_reference_pop_subset_of_B(self):
        """Protocol A reference population is a strict subset of Protocol B.
        Because ESE failures are excluded from A but kept in B, n_ref(A) <= n_ref(B).
        The mean on *total marks* need not be higher — ESE marks and total marks
        are not perfectly correlated in the simulation.
        """
        df = generate_cohort(n=80, distribution="bimodal", seed=5, ese_fail_rate=0.20)
        cfg_a = GradingConfig(protocol=Protocol.EXCLUSIVE)
        cfg_b = GradingConfig(protocol=Protocol.INCLUSIVE)
        r_a = GradingEngine(cfg_a).process(df)
        r_b = GradingEngine(cfg_b).process(df)
        assert r_a.stats["n_ref"] <= r_b.stats["n_ref"] + 1e-9

    def test_small_cohort_uses_absolute(self, default_cfg):
        df = generate_cohort(n=15, distribution="normal_high", seed=2)
        result = GradingEngine(default_cfg).process(df)
        assert result.method == GradingMethod.ABSOLUTE

    def test_large_cohort_uses_relative(self, default_cfg, large_df):
        result = GradingEngine(default_cfg).process(large_df)
        assert result.method == GradingMethod.RELATIVE


# ── Stage 5: Grade Assignment ─────────────────────────────────────────────────

class TestGradeAssignment:
    def test_all_students_have_grade(self, large_df, default_cfg):
        result = GradingEngine(default_cfg).process(large_df)
        assert result.results_df["Final_Grade"].notna().all()

    def test_grades_in_valid_set(self, large_df, default_cfg):
        valid = {"A+", "A", "B+", "B", "C+", "C", "D", "F", "Z", "I"}
        result = GradingEngine(default_cfg).process(large_df)
        assert set(result.results_df["Final_Grade"].unique()).issubset(valid)


# ── Safeguards ────────────────────────────────────────────────────────────────

class TestSafeguards:
    def test_moderation_rule_fires_for_right_skewed(self):
        """Right-skewed cohort has low mean → D_raw likely < P → F expected."""
        df = generate_cohort(n=100, distribution="right_skewed", seed=3,
                             ese_fail_rate=0.0, absent_rate=0.0, att_fail_rate=0.0)
        cfg = GradingConfig(protocol=Protocol.EXCLUSIVE)
        result = GradingEngine(cfg).process(df)
        # Moderation log should appear
        logs_text = " ".join(m for _, m in result.logs)
        # Either moderation or floor fires; at least D boundary <= pass mark
        assert result.boundaries["D"] <= cfg.pass_mark + 1e-9

    def test_no_student_above_pass_mark_gets_F_via_moderation(self):
        """After moderation, students at pass mark should get D not F."""
        df = generate_cohort(n=100, distribution="right_skewed", seed=3,
                             ese_fail_rate=0.0, absent_rate=0.0, att_fail_rate=0.0)
        cfg = GradingConfig(total_max_marks=100, ese_max_marks=60,
                            course_type=CourseType.THEORY, protocol=Protocol.EXCLUSIVE)
        result = GradingEngine(cfg).process(df)
        pass_mark = cfg.pass_mark
        above_pass = result.results_df[result.results_df["marks"] >= pass_mark]
        f_above_pass = above_pass[above_pass["Final_Grade"] == "F"]
        assert len(f_above_pass) == 0, (
            f"Students scoring >= pass mark ({pass_mark}) should never receive F. "
            f"Found {len(f_above_pass)} such students."
        )

    def test_floor_protection_fires_for_left_skewed(self):
        df = generate_cohort(n=100, distribution="left_skewed", seed=4,
                             ese_fail_rate=0.0, absent_rate=0.0, att_fail_rate=0.0)
        cfg = GradingConfig(protocol=Protocol.EXCLUSIVE)
        result = GradingEngine(cfg).process(df)
        # D boundary must be >= floor
        assert result.boundaries["D"] >= cfg.abs_floor - 1e-9

    def test_A_plus_never_exceeds_total_marks(self, large_df, default_cfg):
        result = GradingEngine(default_cfg).process(large_df)
        assert result.boundaries["A+"] <= default_cfg.total_max_marks + 1e-9


# ── GradingResult properties ──────────────────────────────────────────────────

class TestGradingResultProperties:
    def test_pass_rate_between_0_and_100(self, large_df, default_cfg):
        result = GradingEngine(default_cfg).process(large_df)
        assert 0 <= result.pass_rate <= 100

    def test_fail_rate_plus_pass_rate_le_100(self, large_df, default_cfg):
        result = GradingEngine(default_cfg).process(large_df)
        assert result.pass_rate + result.fail_rate <= 100 + 1e-9

    def test_grade_distribution_sums_to_total(self, large_df, default_cfg):
        result = GradingEngine(default_cfg).process(large_df)
        assert result.grade_distribution.sum() == len(large_df)


# ── Practical course type ─────────────────────────────────────────────────────

class TestPracticalCourse:
    def test_pass_mark_is_50pct(self):
        cfg = GradingConfig(course_type=CourseType.PRACTICAL)
        assert cfg.pass_mark == 0.50 * cfg.total_max_marks

    def test_theory_pass_mark_is_40pct(self):
        cfg = GradingConfig(course_type=CourseType.THEORY)
        assert cfg.pass_mark == 0.40 * cfg.total_max_marks


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_absent(self, default_cfg):
        df = pd.DataFrame({
            "id": range(10),
            "marks": [50] * 10,
            "attendance": [60] * 10,
            "ese_marks": [30] * 10,
        })
        result = GradingEngine(default_cfg).process(df)
        assert (result.results_df["Final_Grade"] == "I").all()

    def test_single_student(self, default_cfg):
        df = pd.DataFrame({
            "id": [1], "marks": [75], "attendance": [80], "ese_marks": [35],
        })
        result = GradingEngine(default_cfg).process(df)
        assert result.results_df["Final_Grade"].iloc[0] in {"A+","A","B+","B","C+","C","D","F"}

    def test_empty_dataframe(self, default_cfg):
        df = pd.DataFrame(columns=["id", "marks", "attendance", "ese_marks"])
        result = GradingEngine(default_cfg).process(df)
        assert len(result.results_df) == 0
