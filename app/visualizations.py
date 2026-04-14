"""
visualizations.py
=================
Altair chart builders for the Streamlit dashboard.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import altair as alt
import pandas as pd
import numpy as np

from app.grading_engine import GRADE_ORDER, PASSING_GRADES, GradingResult


# ── Palette ─────────────────────────────────────────────────────────────────
GRADE_COLORS = {
    "A+": "#1a9850", "A":  "#66bd63", "B+": "#a6d96a",
    "B":  "#d9ef8b", "C+": "#fee08b", "C":  "#fdae61",
    "D":  "#f46d43", "F":  "#d73027", "Z":  "#8073ac", "I":  "#bababa",
}


def grade_distribution_chart(result: GradingResult, height: int = 320) -> alt.Chart:
    """Vertical bar chart of grade counts with colour encoding."""
    present_grades = [g for g in GRADE_ORDER if g in result.grade_distribution.index]
    counts = result.grade_distribution.reindex(present_grades).fillna(0).reset_index()
    counts.columns = ["Grade", "Count"]
    counts["Color"] = counts["Grade"].map(GRADE_COLORS)
    counts["PassFail"] = counts["Grade"].apply(
        lambda g: "Pass" if g in PASSING_GRADES else "Fail/Absent"
    )

    bars = (
        alt.Chart(counts)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Grade:N", sort=present_grades, axis=alt.Axis(labelFontSize=13)),
            y=alt.Y("Count:Q", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("Grade:N",
                scale=alt.Scale(domain=list(GRADE_COLORS.keys()),
                                range=list(GRADE_COLORS.values())),
                legend=None),
            tooltip=["Grade", "Count",
                     alt.Tooltip("PassFail:N", title="Status")],
        )
    )
    text = bars.mark_text(dy=-7, fontSize=11, fontWeight="bold").encode(
        text="Count:Q"
    )
    return (bars + text).properties(height=height).configure_view(strokeWidth=0)


def boundary_overlay_chart(
    result: GradingResult,
    height: int = 340,
) -> alt.Chart:
    """Histogram of marks with vertical grade-boundary lines overlaid."""
    df   = result.results_df
    M    = result.config.total_max_marks
    bins = min(30, int(np.sqrt(len(df))))

    hist = (
        alt.Chart(df)
        .mark_bar(opacity=0.65, color="#4c78a8", binSpacing=1)
        .encode(
            x=alt.X("marks:Q", bin=alt.Bin(maxbins=bins),
                    scale=alt.Scale(domain=[0, M]),
                    title="Total Marks"),
            y=alt.Y("count():Q", title="Students"),
            tooltip=["count():Q"],
        )
    )

    boundary_records = [
        {"mark": v, "Grade": k}
        for k, v in result.boundaries.items()
        if 0 <= v <= M
    ]
    bdf = pd.DataFrame(boundary_records)

    if not bdf.empty:
        lines = (
            alt.Chart(bdf)
            .mark_rule(strokeDash=[4, 3], strokeWidth=1.8)
            .encode(
                x="mark:Q",
                color=alt.Color("Grade:N",
                    scale=alt.Scale(domain=list(GRADE_COLORS.keys()),
                                    range=list(GRADE_COLORS.values()))),
                tooltip=["Grade", alt.Tooltip("mark:Q", title="Boundary", format=".1f")],
            )
        )
        labels = (
            alt.Chart(bdf)
            .mark_text(dy=-8, angle=0, fontSize=10, fontWeight="bold")
            .encode(
                x="mark:Q",
                y=alt.value(10),
                text="Grade:N",
                color=alt.Color("Grade:N",
                    scale=alt.Scale(domain=list(GRADE_COLORS.keys()),
                                    range=list(GRADE_COLORS.values())),
                    legend=None),
            )
        )
        chart = (hist + lines + labels)
    else:
        chart = hist

    return chart.properties(height=height).configure_view(strokeWidth=0)


def protocol_comparison_chart(
    result_a: GradingResult,
    result_b: GradingResult,
    height: int = 300,
) -> alt.Chart:
    """Side-by-side bar chart comparing grade distributions for A vs B."""
    def _to_df(r: GradingResult, label: str) -> pd.DataFrame:
        present = [g for g in GRADE_ORDER if g in r.grade_distribution.index]
        s = r.grade_distribution.reindex(present).fillna(0)
        total = len(r.results_df)
        out = s.reset_index()
        out.columns = ["Grade", "Count"]
        out["Pct"]      = (out["Count"] / total * 100).round(1)
        out["Protocol"] = label
        return out

    combined = pd.concat([
        _to_df(result_a, "Protocol A (Exclusive)"),
        _to_df(result_b, "Protocol B (Inclusive)"),
    ], ignore_index=True)

    return (
        alt.Chart(combined)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Grade:N", sort=GRADE_ORDER),
            y=alt.Y("Pct:Q", title="Students (%)"),
            color=alt.Color("Protocol:N",
                scale=alt.Scale(range=["#2171b5", "#cb181d"]),
                legend=alt.Legend(orient="top")),
            xOffset="Protocol:N",
            tooltip=["Grade", "Protocol",
                     alt.Tooltip("Pct:Q", title="%", format=".1f"),
                     "Count"],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
    )


def scatter_marks_chart(result: GradingResult, height: int = 320) -> alt.Chart:
    """Scatter plot: total marks vs ESE marks, coloured by final grade."""
    df = result.results_df.copy()
    df["_ese_num"] = pd.to_numeric(df["ese_marks"], errors="coerce")

    return (
        alt.Chart(df.dropna(subset=["_ese_num"]))
        .mark_circle(size=55, opacity=0.75)
        .encode(
            x=alt.X("marks:Q", title="Total Marks",
                    scale=alt.Scale(domain=[0, result.config.total_max_marks])),
            y=alt.Y("_ese_num:Q", title="ESE Marks",
                    scale=alt.Scale(domain=[0, result.config.ese_max_marks])),
            color=alt.Color("Final_Grade:N",
                scale=alt.Scale(domain=list(GRADE_COLORS.keys()),
                                range=list(GRADE_COLORS.values())),
                legend=alt.Legend(title="Grade")),
            tooltip=["id", "marks", "_ese_num", "attendance", "Final_Grade"],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
    )


def boundaries_table_df(boundaries: Dict[str, float]) -> pd.DataFrame:
    rows = []
    grade_list = ["A+", "A", "B+", "B", "C+", "C", "D"]
    for i, g in enumerate(grade_list):
        low = boundaries.get(g, 0)
        high = boundaries.get(grade_list[i - 1], None) if i > 0 else None
        rows.append({
            "Grade": g,
            "Min Marks": f"{low:.1f}",
            "Range": f"{low:.1f} – {(str(round(high, 1)) + ' ') if high else 'Max'}",
            "Status": "Pass" if g in PASSING_GRADES else "Fail",
        })
    rows.append({"Grade": "F", "Min Marks": "0", "Range": f"0 – {boundaries.get('D', 0):.1f}", "Status": "Fail"})
    return pd.DataFrame(rows)
