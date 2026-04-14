"""
simulation.py
=============
Synthetic cohort generators used in Section VII of the companion paper.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


DISTRIBUTIONS = {
    "Normal – High Performers":   "normal_high",
    "Normal – Low Performers":    "normal_low",
    "Bimodal (30% weak tail)":    "bimodal",
    "Left-Skewed (Beta 8,2)":     "left_skewed",
    "Right-Skewed (Beta 2,6)":    "right_skewed",
    "Uniform":                    "uniform",
}


def generate_cohort(
    n:              int   = 100,
    distribution:   str   = "normal_high",
    ese_fail_rate:  float = 0.15,
    absent_rate:    float = 0.05,
    att_fail_rate:  float = 0.05,
    total_marks:    int   = 100,
    ese_max_marks:  int   = 60,
    seed:           Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic student cohort for testing the grading engine.

    Parameters
    ----------
    n               : Number of students.
    distribution    : One of the keys in DISTRIBUTIONS or a short-code value.
    ese_fail_rate   : Fraction of eligible students who score below ESE minimum.
    absent_rate     : Fraction of eligible students marked absent in ESE.
    att_fail_rate   : Fraction of students below 75% attendance.
    total_marks     : Total course marks.
    ese_max_marks   : Maximum ESE marks.
    seed            : RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate raw marks
    marks = _draw_marks(n, distribution, total_marks, rng)
    marks = np.clip(np.round(marks, 1), 0, total_marks)

    # 2. Attendance
    attendance = rng.uniform(50, 100, n)
    att_fail_idx = rng.choice(n, size=int(n * att_fail_rate), replace=False)
    attendance[att_fail_idx] = rng.uniform(40, 74.9, len(att_fail_idx))
    attendance = np.clip(np.round(attendance, 1), 0, 100)

    # 3. ESE marks  (set proportionally to total marks for non-failing students)
    ese_marks: list = []
    ese_min = 0.20 * ese_max_marks
    for i in range(n):
        if attendance[i] < 75:
            ese_marks.append(round(float(marks[i]) * ese_max_marks / total_marks, 1))
            continue
        r = rng.random()
        if r < absent_rate:
            ese_marks.append("AB")
        elif r < absent_rate + ese_fail_rate:
            ese_marks.append(round(rng.uniform(0, ese_min - 0.1), 1))
        else:
            # proportional + small noise
            base = float(marks[i]) * ese_max_marks / total_marks
            noisy = rng.normal(base, 3)
            ese_marks.append(round(float(np.clip(noisy, ese_min, ese_max_marks)), 1))

    return pd.DataFrame({
        "id":          range(1, n + 1),
        "marks":       marks,
        "attendance":  attendance,
        "ese_marks":   ese_marks,
    })


def _draw_marks(n: int, dist: str, M: int, rng: np.random.Generator) -> np.ndarray:
    if dist == "normal_high":
        return rng.normal(0.72 * M, 0.10 * M, n)
    if dist == "normal_low":
        return rng.normal(0.48 * M, 0.12 * M, n)
    if dist == "bimodal":
        k = int(0.70 * n)
        return np.concatenate([
            rng.normal(0.75 * M, 0.08 * M, k),
            rng.normal(0.30 * M, 0.06 * M, n - k),
        ])
    if dist == "left_skewed":
        return rng.beta(8, 2, n) * M
    if dist == "right_skewed":
        return rng.beta(2, 6, n) * M
    if dist == "uniform":
        return rng.uniform(0.20 * M, M, n)
    # fallback
    return rng.normal(0.65 * M, 0.12 * M, n)
