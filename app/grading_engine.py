"""
grading_engine.py
=================
Core grading logic for the StrictUniversityGrading framework.

Published in:
  D. Mandal, "Automated Relative Grading With Dual Statistical Protocols:
  A Transparent, Rule-Based Framework for Higher-Education Assessment,"
  IEEE Transactions on Education, 2026.

Repository : https://github.com/dmandal/strict-university-grading
License    : MIT
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

GRADE_ORDER: List[str] = ["A+", "A", "B+", "B", "C+", "C", "D", "F", "Z", "I"]
"""Ordered list of all possible grade symbols (best → worst)."""

PASSING_GRADES: List[str] = ["A+", "A", "B+", "B", "C+", "C", "D"]
"""Grade symbols that count as a pass."""

TERMINAL_SYMBOLS: Dict[str, str] = {
    "I": "Ineligible – attendance below threshold",
    "Z": "Absent – did not appear in ESE",
    "F": "Failed – below ESE minimum or total pass mark",
}


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CourseType(str, Enum):
    THEORY     = "Theory"
    PRACTICAL  = "Practical"


class Protocol(str, Enum):
    EXCLUSIVE = "Protocol A (Exclusive)"
    INCLUSIVE = "Protocol B (Inclusive)"


class GradingMethod(str, Enum):
    RELATIVE = "Relative Grading (Statistical)"
    ABSOLUTE = "Absolute Grading (Batch < 30)"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GradingConfig:
    """Immutable configuration for a single grading run."""
    total_max_marks:  float      = 100.0
    ese_max_marks:    float      = 60.0
    course_type:      CourseType = CourseType.THEORY
    protocol:         Protocol   = Protocol.EXCLUSIVE
    attendance_floor: float      = 75.0   # percent
    ese_min_fraction: float      = 0.20   # fraction of ese_max_marks
    abs_cutoff_frac:  float      = 0.30   # fraction of total_max_marks
    min_n_relative:   int        = 30     # cohort threshold for relative grading

    @property
    def pass_mark(self) -> float:
        fraction = 0.50 if self.course_type == CourseType.PRACTICAL else 0.40
        return fraction * self.total_max_marks

    @property
    def ese_threshold(self) -> float:
        return self.ese_min_fraction * self.ese_max_marks

    @property
    def abs_floor(self) -> float:
        return self.abs_cutoff_frac * self.total_max_marks


@dataclasses.dataclass
class GradingResult:
    """Container returned by :func:`GradingEngine.process`."""
    results_df:    pd.DataFrame
    boundaries:    Dict[str, float]
    method:        GradingMethod
    stats:         Dict[str, float]          # mean, sigma, n_ref, n_total
    logs:          List[Tuple[str, str]]     # (level, message)  level ∈ {info,warn,error}
    config:        GradingConfig

    # Convenience properties ------------------------------------------------

    @property
    def pass_rate(self) -> float:
        total = len(self.results_df)
        if total == 0:
            return 0.0
        passing = self.results_df["Final_Grade"].isin(PASSING_GRADES).sum()
        return passing / total * 100

    @property
    def grade_distribution(self) -> pd.Series:
        return self.results_df["Final_Grade"].value_counts()

    @property
    def top_grade_rate(self) -> float:
        total = len(self.results_df)
        if total == 0:
            return 0.0
        top = self.results_df["Final_Grade"].isin(["A+", "A"]).sum()
        return top / total * 100

    @property
    def fail_rate(self) -> float:
        total = len(self.results_df)
        if total == 0:
            return 0.0
        failed = self.results_df["Final_Grade"].isin(["F", "I", "Z"]).sum()
        return failed / total * 100


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class GradingEngine:
    """
    Rule-based, auditable grading engine implementing the pipeline described
    in the companion IEEE Transactions on Education paper.

    Parameters
    ----------
    config : GradingConfig
        Immutable run configuration.

    Examples
    --------
    >>> import pandas as pd
    >>> from app.grading_engine import GradingEngine, GradingConfig, CourseType, Protocol
    >>> df = pd.read_csv("data/sample_cohort.csv")
    >>> cfg = GradingConfig(course_type=CourseType.THEORY, protocol=Protocol.EXCLUSIVE)
    >>> engine = GradingEngine(cfg)
    >>> result = engine.process(df)
    >>> print(result.pass_rate)
    """

    def __init__(self, config: GradingConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, df: pd.DataFrame) -> GradingResult:
        """
        Execute the full five-stage grading pipeline on *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: ``id``, ``marks``, ``attendance``, ``ese_marks``.

        Returns
        -------
        GradingResult
        """
        cfg  = self.config
        logs: List[Tuple[str, str]] = []
        data = df.copy()

        # ── Stage 1: Attendance gate ────────────────────────────────────
        mask_absent_att = data["attendance"] < cfg.attendance_floor
        n_att = mask_absent_att.sum()
        data["Final_Grade"] = np.where(mask_absent_att, "I", None)
        if n_att:
            logs.append(("error",
                f"{n_att} student(s) marked 'I' — attendance < {cfg.attendance_floor:.0f}%."))

        # ── Stage 2: ESE absentee check ─────────────────────────────────
        mask_ese_ab = (
            data["ese_marks"].astype(str).str.strip().str.upper() == "AB"
        ) & data["Final_Grade"].isna()
        n_ab = mask_ese_ab.sum()
        data.loc[mask_ese_ab, "Final_Grade"] = "Z"
        if n_ab:
            logs.append(("error",
                f"{n_ab} student(s) marked 'Z' — absent from ESE."))

        # ── Stage 3: ESE minimum marks ──────────────────────────────────
        data["_ese_num"] = pd.to_numeric(data["ese_marks"], errors="coerce").fillna(0)
        mask_ese_fail = (
            data["Final_Grade"].isna() &
            (data["_ese_num"] < cfg.ese_threshold)
        )
        n_ese_fail = mask_ese_fail.sum()
        data.loc[mask_ese_fail, "Final_Grade"] = "F"
        if n_ese_fail:
            logs.append(("error",
                f"{n_ese_fail} student(s) assigned 'F' — ESE score "
                f"< {cfg.ese_threshold:.1f} ({cfg.ese_min_fraction*100:.0f}% of "
                f"{cfg.ese_max_marks:.0f})."))

        # ── Stage 4: Protocol selection & boundary derivation ───────────
        ref_pop = self._build_reference_population(data)
        n_ref   = len(ref_pop)
        logs.append(("info",
            f"Protocol: {cfg.protocol.value}  |  "
            f"Reference population: {n_ref} student(s)."))

        if n_ref >= cfg.min_n_relative:
            method = GradingMethod.RELATIVE
            boundaries, b_logs, stats = self._relative_boundaries(ref_pop["marks"].values)
            logs.extend(b_logs)
        else:
            method = GradingMethod.ABSOLUTE
            boundaries = self._absolute_boundaries()
            stats = {"mean": float(np.mean(ref_pop["marks"].values)) if n_ref else 0.0,
                     "sigma": float(np.std(ref_pop["marks"].values)) if n_ref else 0.0,
                     "n_ref": float(n_ref),
                     "n_total": float(len(data))}
            logs.append(("warn",
                f"Cohort size {n_ref} < {cfg.min_n_relative}. "
                f"Switched to absolute grading."))

        stats["n_ref"]   = float(n_ref)
        stats["n_total"] = float(len(data))

        # ── Stage 5: Grade assignment ────────────────────────────────────
        eligible = data["Final_Grade"].isna()
        data.loc[eligible, "Final_Grade"] = data.loc[eligible, "marks"].apply(
            lambda x: self._assign_grade(x, boundaries)
        )

        data = data.drop(columns=["_ese_num"])
        logs.append(("info", f"Grading complete. Method: {method.value}."))

        return GradingResult(
            results_df = data,
            boundaries = boundaries,
            method     = method,
            stats      = stats,
            logs       = logs,
            config     = cfg,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_reference_population(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if cfg.protocol == Protocol.EXCLUSIVE:
            mask = (
                (data["attendance"] >= cfg.attendance_floor) &
                (data["_ese_num"] >= cfg.ese_threshold)
            )
        else:  # INCLUSIVE
            mask = data["attendance"] >= cfg.attendance_floor
        return data.loc[mask]

    def _relative_boundaries(
        self, marks: np.ndarray
    ) -> Tuple[Dict[str, float], List[Tuple[str, str]], Dict[str, float]]:
        cfg  = self.config
        logs: List[Tuple[str, str]] = []
        X    = float(np.mean(marks))
        sig  = float(np.std(marks))
        logs.append(("info",
            f"Batch statistics — Mean X̄ = {X:.3f},  σ = {sig:.3f}"))

        bounds: Dict[str, float] = {
            "A+": X + 1.5 * sig,
            "A":  X + 1.0 * sig,
            "B+": X + 0.5 * sig,
            "B":  X,
            "C+": X - 0.5 * sig,
            "C":  X - 1.0 * sig,
            "D":  X - 1.5 * sig,
        }
        D_raw = bounds["D"]

        # Safeguard 1 — Moderation rule
        if D_raw > cfg.pass_mark:
            logs.append(("warn",
                f"Moderation rule triggered: D_raw ({D_raw:.2f}) > P ({cfg.pass_mark:.2f}). "
                f"Lower spine collapsed to pass mark."))
            bounds["C+"] = X - (1 / 3) * (X - cfg.pass_mark)
            bounds["C"]  = X - (2 / 3) * (X - cfg.pass_mark)
            bounds["D"]  = float(cfg.pass_mark)

        # Safeguard 2 — Minimum cutoff floor
        elif D_raw < cfg.abs_floor:
            delta = cfg.abs_floor - D_raw
            logs.append(("warn",
                f"Floor-protection triggered: D_raw ({D_raw:.2f}) < ϕ ({cfg.abs_floor:.2f}). "
                f"Curve shifted up by Δ = {delta:.2f}."))
            bounds = {g: v + delta for g, v in bounds.items()}
            bounds["D"] = cfg.abs_floor

        # Upper-bound cap
        if bounds["A+"] > cfg.total_max_marks:
            bounds["A+"] = float(cfg.total_max_marks)
            logs.append(("info",
                f"A+ upper-bound capped at {cfg.total_max_marks:.0f}."))

        stats = {"mean": X, "sigma": sig, "n_ref": 0.0, "n_total": 0.0}
        return bounds, logs, stats

    def _absolute_boundaries(self) -> Dict[str, float]:
        if self.config.course_type == CourseType.THEORY:
            return {"A+": 90, "A": 80, "B+": 72, "B": 64, "C+": 56, "C": 48, "D": 40}
        else:  # Practical
            return {"A+": 90, "A": 80, "B+": 70, "B": 62, "C+": 58, "C": 54, "D": 50}

    @staticmethod
    def _assign_grade(marks: float, bounds: Dict[str, float]) -> str:
        for grade in ["A+", "A", "B+", "B", "C+", "C", "D"]:
            if marks >= bounds[grade]:
                return grade
        return "F"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_grading(
    df: pd.DataFrame,
    total_max_marks: float  = 100.0,
    ese_max_marks:   float  = 60.0,
    course_type:     str    = "Theory",
    protocol:        str    = "Protocol A (Exclusive)",
) -> GradingResult:
    """One-line helper for scripting / testing."""
    cfg = GradingConfig(
        total_max_marks = total_max_marks,
        ese_max_marks   = ese_max_marks,
        course_type     = CourseType(course_type),
        protocol        = Protocol(protocol),
    )
    return GradingEngine(cfg).process(df)
