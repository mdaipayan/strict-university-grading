"""
main.py  ·  StrictUniversityGrading — Interactive Dashboard
============================================================
Companion web application for:

  D. Mandal, "Automated Relative Grading With Dual Statistical Protocols:
  A Transparent, Rule-Based Framework for Higher-Education Assessment,"
  IEEE Transactions on Education, 2026.

Run:
    streamlit run main.py
"""

import io
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from app.grading_engine import (
    GradingConfig, GradingEngine, GradingResult,
    CourseType, Protocol, GradingMethod,
    GRADE_ORDER, PASSING_GRADES, TERMINAL_SYMBOLS,
)
from app.simulation import generate_cohort, DISTRIBUTIONS
from app.visualizations import (
    grade_distribution_chart,
    boundary_overlay_chart,
    protocol_comparison_chart,
    scatter_marks_chart,
    boundaries_table_df,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Realative Grading Tool",
    page_icon   = "🎓",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
code, pre, .stCode {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Header strip ── */
.header-strip {
    background: linear-gradient(135deg, #0f2042 0%, #1a3a6e 60%, #1565c0 100%);
    border-radius: 10px;
    padding: 24px 32px 20px;
    margin-bottom: 18px;
    color: white;
}
.header-strip h1 { color: white; margin: 0 0 4px; font-size: 1.9rem; font-weight: 700; }
.header-strip p  { color: #b0c4de; margin: 0; font-size: 0.85rem; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #f8faff;
    border: 1px solid #dde4f0;
    border-radius: 8px;
    padding: 14px 16px;
}
div[data-testid="metric-container"] label {
    font-size: 0.75rem !important;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #5a6e8c !important;
}

/* ── Section headings ── */
.section-label {
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #1565c0;
    margin: 18px 0 6px;
    padding-left: 2px;
}

/* ── Log styles ── */
.log-info  { background:#e8f0fe; border-left:3px solid #1565c0; padding:6px 10px; border-radius:4px; margin:3px 0; font-size:0.83rem; }
.log-warn  { background:#fff3cd; border-left:3px solid #f5a623; padding:6px 10px; border-radius:4px; margin:3px 0; font-size:0.83rem; }
.log-error { background:#fde8e8; border-left:3px solid #c62828; padding:6px 10px; border-radius:4px; margin:3px 0; font-size:0.83rem; }

/* ── Grade chip ── */
.grade-chip {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-weight:700; font-size:0.78rem; font-family:'IBM Plex Mono',monospace;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #f0f4ff; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.88rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 0.87rem;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Configuration panel
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    course_type_str = st.radio("Course Type", ["Theory", "Practical"], horizontal=True)
    course_type = CourseType(course_type_str)

    protocol_str = st.radio(
        "Statistical Protocol",
        [Protocol.EXCLUSIVE.value, Protocol.INCLUSIVE.value],
        help=(
            "**Protocol A (Exclusive):** Excludes ESE failures from reference "
            "population → prevents zero-inflation bias.\n\n"
            "**Protocol B (Inclusive):** Retains ESE failures in reference "
            "population → lowers mean, benefits borderline passers."
        ),
    )
    protocol = Protocol(protocol_str)

    st.markdown("---")
    total_marks  = st.number_input("Total Course Marks (M)", value=100, min_value=10, max_value=1000)
    ese_max      = st.number_input("ESE Maximum Marks",       value=int(total_marks * 0.60),
                                   min_value=1, max_value=int(total_marks))

    # Derived values
    P     = 0.50 * total_marks if course_type == CourseType.PRACTICAL else 0.40 * total_marks
    theta = 0.20 * ese_max
    phi   = 0.30 * total_marks

    st.markdown("---")
    st.markdown("**Rules Summary**")
    st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Pass Mark (P) | {P:.0f} |
| ESE Min (θ) | {theta:.1f} |
| Floor (ϕ) | {phi:.0f} |
| Protocol | {'A' if 'Exclusive' in protocol_str else 'B'} |
""")

    st.markdown("---")
    st.markdown("**Reference**")
    st.caption(
        "D. Mandal, *IEEE Trans. Educ.*, 2026.  \n"
        "[GitHub Repository](https://github.com/dmandal/strict-university-grading)"
    )


# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-strip">
  <h1>🎓 StrictUniversityGrading</h1>
  <p>Automated Relative Grading with Dual Statistical Protocols · IEEE Transactions on Education (2026)</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════

tab_grade, tab_compare, tab_simulate, tab_about = st.tabs([
    "📊 Grade Calculator",
    "⚖️ Protocol Comparison",
    "🔬 Simulation Lab",
    "📄 About & Citation",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Grade Calculator
# ────────────────────────────────────────────────────────────────────────────

with tab_grade:
    cfg = GradingConfig(
        total_max_marks = float(total_marks),
        ese_max_marks   = float(ese_max),
        course_type     = course_type,
        protocol        = protocol,
    )

    # ── Data source ──────────────────────────────────────────────────────
    data_src = st.radio(
        "Data source",
        ["Upload CSV", "Use sample data"],
        horizontal=True,
        key="data_src_tab1",
    )

    df_input: pd.DataFrame | None = None

    if data_src == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload student CSV  (columns: id, marks, attendance, ese_marks)",
            type=["csv"], key="upload_tab1",
        )
        if uploaded:
            try:
                df_raw = pd.read_csv(uploaded)
                df_raw.columns = df_raw.columns.str.strip().str.lower()
                required = {"id", "marks", "attendance", "ese_marks"}
                missing  = required - set(df_raw.columns)
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    df_input = df_raw
            except Exception as exc:
                st.error(f"Could not read file: {exc}")
    else:
        dist_label = st.selectbox(
            "Synthetic distribution", list(DISTRIBUTIONS.keys()), index=0,
        )
        n_students = st.slider("Cohort size", 20, 500, 100, 10)
        seed_val   = st.number_input("Random seed", value=42)
        df_input   = generate_cohort(
            n            = n_students,
            distribution = DISTRIBUTIONS[dist_label],
            seed         = int(seed_val),
            total_marks  = int(total_marks),
            ese_max_marks= int(ese_max),
        )
        st.caption(f"Generated {n_students} synthetic students — {dist_label}")

    # ── Download CSV template ─────────────────────────────────────────────
    sample = pd.DataFrame({
        "id": range(1, 6),
        "marks": [82, 65, 45, 32, 91],
        "attendance": [90, 85, 80, 76, 95],
        "ese_marks": [40, 30, "AB", 10, 50],
    })
    st.download_button(
        "⬇ Download CSV template",
        sample.to_csv(index=False).encode(),
        "template.csv", "text/csv",
        key="dl_template",
    )

    # ── Run grading ────────────────────────────────────────────────────────
    if df_input is not None:
        try:
            engine = GradingEngine(cfg)
            result = engine.process(df_input)

            # ── KPI strip ────────────────────────────────────────────────
            k1, k2, k3, k4, k5 = st.columns(5)
            eligible = df_input[df_input["attendance"] >= 75]
            k1.metric("Total Students",    len(df_input))
            k2.metric("Mean Marks (eligible)", f"{eligible['marks'].mean():.1f}" if len(eligible) else "—")
            k3.metric("Method",
                      "Relative" if result.method == GradingMethod.RELATIVE else "Absolute")
            k4.metric("Pass Rate",  f"{result.pass_rate:.1f}%")
            k5.metric("A / A+ Rate", f"{result.top_grade_rate:.1f}%")

            # ── Logs ─────────────────────────────────────────────────────
            with st.expander("📋 Audit Log", expanded=True):
                for level, msg in result.logs:
                    cls = {"info": "log-info", "warn": "log-warn", "error": "log-error"}[level]
                    st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

            # ── Charts ───────────────────────────────────────────────────
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown('<p class="section-label">Grade Distribution</p>',
                            unsafe_allow_html=True)
                st.altair_chart(grade_distribution_chart(result), use_container_width=True)

                st.markdown('<p class="section-label">Grade Boundaries</p>',
                            unsafe_allow_html=True)
                bdf = boundaries_table_df(result.boundaries)
                st.dataframe(
                    bdf.style.apply(
                        lambda r: ["background-color:#e8f5e9" if r["Status"] == "Pass"
                                   else "background-color:#fde8e8"] * len(r), axis=1
                    ),
                    hide_index=True, use_container_width=True,
                )

            with c2:
                st.markdown('<p class="section-label">Marks Distribution + Boundaries</p>',
                            unsafe_allow_html=True)
                st.altair_chart(boundary_overlay_chart(result), use_container_width=True)

                st.markdown('<p class="section-label">Total vs ESE Marks</p>',
                            unsafe_allow_html=True)
                st.altair_chart(scatter_marks_chart(result), use_container_width=True)

            # ── Results table ─────────────────────────────────────────────
            st.markdown('<p class="section-label">Student Results</p>',
                        unsafe_allow_html=True)

            def _highlight(row: pd.Series):
                g = row.get("Final_Grade", "")
                if g in ("F", "I", "Z"):  return ["background-color:#fde8e8"] * len(row)
                if g in ("A+", "A"):      return ["background-color:#e8f5e9"] * len(row)
                return [""] * len(row)

            st.dataframe(
                result.results_df.style.apply(_highlight, axis=1),
                use_container_width=True, height=300,
            )

            # ── Download ──────────────────────────────────────────────────
            res_csv = result.results_df.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download Results CSV",
                res_csv, "final_grades.csv", "text/csv",
            )

            # ── Text report ───────────────────────────────────────────────
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            report_lines = [
                "=" * 60,
                " StrictUniversityGrading — Audit Report",
                f" Generated : {ts}",
                "=" * 60,
                f" Course type   : {course_type.value}",
                f" Protocol      : {protocol.value}",
                f" Total marks   : {total_marks}",
                f" ESE max marks : {ese_max}",
                f" Method used   : {result.method.value}",
                "-" * 60,
                " GRADE BOUNDARIES",
            ]
            for g, v in result.boundaries.items():
                report_lines.append(f"   {g:>3s}  ≥  {v:>6.2f}")
            report_lines += [
                "-" * 60,
                " STATISTICS",
                f"   Mean  : {result.stats.get('mean', 0):.3f}",
                f"   Sigma : {result.stats.get('sigma', 0):.3f}",
                f"   N ref : {result.stats.get('n_ref', 0):.0f}",
                "-" * 60,
                " GRADE DISTRIBUTION",
            ]
            for g in GRADE_ORDER:
                cnt = result.grade_distribution.get(g, 0)
                if cnt:
                    report_lines.append(f"   {g:>3s}  :  {cnt}")
            report_lines += [
                "-" * 60,
                f" Pass rate   : {result.pass_rate:.1f}%",
                f" A/A+ rate   : {result.top_grade_rate:.1f}%",
                f" Fail/absent : {result.fail_rate:.1f}%",
                "=" * 60,
                " LOG",
            ]
            for level, msg in result.logs:
                report_lines.append(f"  [{level.upper():5s}] {msg}")
            report_lines.append("=" * 60)
            report_text = "\n".join(report_lines)

            st.download_button(
                "⬇ Download Audit Report (.txt)",
                report_text.encode(),
                "audit_report.txt", "text/plain",
                key="dl_report",
            )

        except Exception as exc:
            st.error(f"Error during grading: {exc}")
            st.exception(exc)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Protocol Comparison
# ────────────────────────────────────────────────────────────────────────────

with tab_compare:
    st.markdown("### Compare Protocol A vs Protocol B on the same cohort")
    st.markdown(
        "Upload or generate a cohort once, then inspect how the choice of "
        "reference population changes grade boundaries and the resulting "
        "distribution — the core empirical finding of Section VII of the paper."
    )

    data_src_c = st.radio(
        "Data source", ["Synthetic data", "Upload CSV"],
        horizontal=True, key="data_src_compare",
    )

    df_cmp: pd.DataFrame | None = None

    if data_src_c == "Upload CSV":
        up_c = st.file_uploader("Upload CSV", type=["csv"], key="ul_cmp")
        if up_c:
            df_c = pd.read_csv(up_c)
            df_c.columns = df_c.columns.str.strip().str.lower()
            df_cmp = df_c
    else:
        c_dist  = st.selectbox("Distribution", list(DISTRIBUTIONS.keys()), key="c_dist")
        c_n     = st.slider("Cohort size", 30, 500, 100, 10, key="c_n")
        c_seed  = st.number_input("Seed", value=99, key="c_seed")
        df_cmp  = generate_cohort(
            n=c_n, distribution=DISTRIBUTIONS[c_dist],
            seed=int(c_seed), total_marks=int(total_marks),
            ese_max_marks=int(ese_max),
        )

    if df_cmp is not None:
        cfg_a = GradingConfig(float(total_marks), float(ese_max), course_type, Protocol.EXCLUSIVE)
        cfg_b = GradingConfig(float(total_marks), float(ese_max), course_type, Protocol.INCLUSIVE)
        r_a   = GradingEngine(cfg_a).process(df_cmp)
        r_b   = GradingEngine(cfg_b).process(df_cmp)

        # KPI comparison
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Protocol A — Exclusive")
            st.metric("Pass Rate",   f"{r_a.pass_rate:.1f}%")
            st.metric("A / A+ Rate", f"{r_a.top_grade_rate:.1f}%")
            st.metric("Fail / Absent", f"{r_a.fail_rate:.1f}%")
            st.metric("Mean (ref)",  f"{r_a.stats.get('mean', 0):.2f}")
            st.metric("σ (ref)",     f"{r_a.stats.get('sigma', 0):.2f}")
        with col_b:
            st.markdown("#### Protocol B — Inclusive")
            st.metric("Pass Rate",   f"{r_b.pass_rate:.1f}%",
                      delta=f"{r_b.pass_rate - r_a.pass_rate:+.1f}%")
            st.metric("A / A+ Rate", f"{r_b.top_grade_rate:.1f}%",
                      delta=f"{r_b.top_grade_rate - r_a.top_grade_rate:+.1f}%")
            st.metric("Fail / Absent", f"{r_b.fail_rate:.1f}%",
                      delta=f"{r_b.fail_rate - r_a.fail_rate:+.1f}%")
            st.metric("Mean (ref)",  f"{r_b.stats.get('mean', 0):.2f}",
                      delta=f"{r_b.stats.get('mean', 0) - r_a.stats.get('mean', 0):+.2f}")
            st.metric("σ (ref)",     f"{r_b.stats.get('sigma', 0):.2f}",
                      delta=f"{r_b.stats.get('sigma', 0) - r_a.stats.get('sigma', 0):+.2f}")

        st.markdown("---")
        st.markdown("#### Side-by-Side Grade Distribution")
        st.altair_chart(
            protocol_comparison_chart(r_a, r_b), use_container_width=True,
        )

        st.markdown("#### Boundary Shift (A → B)")
        ba = r_a.boundaries
        bb = r_b.boundaries
        delta_rows = []
        for g in ["A+", "A", "B+", "B", "C+", "C", "D"]:
            delta_rows.append({
                "Grade": g,
                "Boundary A": f"{ba.get(g, 0):.2f}",
                "Boundary B": f"{bb.get(g, 0):.2f}",
                "Δ (B − A)": f"{bb.get(g, 0) - ba.get(g, 0):+.2f}",
            })
        st.dataframe(pd.DataFrame(delta_rows), hide_index=True, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — Simulation Lab
# ────────────────────────────────────────────────────────────────────────────

with tab_simulate:
    st.markdown("### Monte Carlo Simulation")
    st.markdown(
        "Reproduce the simulation experiments from **Section VII** of the paper.  "
        "Select a distribution and a range of cohort sizes; the engine runs *R* "
        "replications at each size and reports mean pass rate, A/A+ rate, and "
        "boundary standard deviation."
    )

    sc1, sc2, sc3 = st.columns(3)
    sim_dist  = sc1.selectbox("Distribution", list(DISTRIBUTIONS.keys()), key="sim_dist")
    sim_reps  = sc2.slider("Replications (R)", 10, 200, 50, 10)
    sim_sizes = sc3.multiselect(
        "Cohort sizes",
        [20, 30, 50, 60, 100, 150, 200, 300, 500],
        default=[30, 60, 100, 200],
    )
    sim_proto = st.radio(
        "Protocol", [Protocol.EXCLUSIVE.value, Protocol.INCLUSIVE.value],
        horizontal=True, key="sim_proto",
    )

    if st.button("▶ Run Simulation", type="primary"):
        if not sim_sizes:
            st.warning("Select at least one cohort size.")
        else:
            rows = []
            prog = st.progress(0, text="Running…")
            total_jobs = len(sim_sizes) * sim_reps
            job = 0

            for n_s in sim_sizes:
                pass_rates, top_rates, d_bounds = [], [], []
                for rep in range(sim_reps):
                    df_s = generate_cohort(
                        n=n_s,
                        distribution=DISTRIBUTIONS[sim_dist],
                        seed=rep,
                        total_marks=int(total_marks),
                        ese_max_marks=int(ese_max),
                    )
                    cfg_s = GradingConfig(
                        float(total_marks), float(ese_max),
                        course_type, Protocol(sim_proto),
                    )
                    r_s = GradingEngine(cfg_s).process(df_s)
                    pass_rates.append(r_s.pass_rate)
                    top_rates.append(r_s.top_grade_rate)
                    d_bounds.append(r_s.boundaries.get("D", 0))
                    job += 1
                    prog.progress(job / total_jobs, text=f"n={n_s}, rep {rep+1}/{sim_reps}")

                rows.append({
                    "Cohort Size (n)": n_s,
                    "Mean Pass Rate (%)": round(float(np.mean(pass_rates)), 2),
                    "SD Pass Rate": round(float(np.std(pass_rates)), 2),
                    "Mean A/A+ Rate (%)": round(float(np.mean(top_rates)), 2),
                    "Mean D-Boundary": round(float(np.mean(d_bounds)), 2),
                    "SD D-Boundary": round(float(np.std(d_bounds)), 2),
                })

            prog.empty()
            sim_df = pd.DataFrame(rows)
            st.dataframe(sim_df, hide_index=True, use_container_width=True)

            # Chart
            import altair as alt
            chart_pass = (
                alt.Chart(sim_df)
                .mark_line(point=True, color="#1565c0")
                .encode(
                    x=alt.X("Cohort Size (n):Q"),
                    y=alt.Y("Mean Pass Rate (%):Q", scale=alt.Scale(zero=False)),
                    tooltip=["Cohort Size (n)", "Mean Pass Rate (%)", "SD Pass Rate"],
                )
                .properties(title="Pass Rate vs Cohort Size", height=280)
            )
            st.altair_chart(chart_pass, use_container_width=True)

            st.download_button(
                "⬇ Download Simulation Results",
                sim_df.to_csv(index=False).encode(),
                "simulation_results.csv", "text/csv",
            )


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — About & Citation
# ────────────────────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("### About This Application")

    st.markdown("""
This web application is the official companion software for the following paper:

> D. Mandal, **"Automated Relative Grading With Dual Statistical Protocols:
> A Transparent, Rule-Based Framework for Higher-Education Assessment,"**
> *IEEE Transactions on Education*, 2026.

The application implements the complete five-stage grading pipeline described in the paper,
including dual statistical protocols (Protocol A — Exclusive and Protocol B — Inclusive),
the moderation rule (Safeguard 1), and the minimum cutoff floor (Safeguard 2).
    """)

    with st.expander("📐 Algorithm Overview"):
        st.markdown("""
| Stage | Rule | Terminal Symbol |
|-------|------|-----------------|
| 1 | Attendance < 75% | **I** — Ineligible |
| 2 | ESE = "AB" | **Z** — Absent |
| 3 | ESE score < 20% of ESE max | **F** — Failed ESE |
| 4 | Protocol selection & boundary derivation | — |
| 5 | Grade assignment via boundary lookup | A+ / A / B+ / B / C+ / C / D / F |

**Boundary formula (n ≥ 30):**

| Grade | Lower Boundary |
|-------|---------------|
| A+  | X̄ + 1.5 σ |
| A   | X̄ + 1.0 σ |
| B+  | X̄ + 0.5 σ |
| B   | X̄            |
| C+  | X̄ − 0.5 σ |
| C   | X̄ − 1.0 σ |
| D   | X̄ − 1.5 σ  (subject to safeguards) |
        """)

    with st.expander("📖 BibTeX Citation"):
        bibtex = textwrap.dedent("""\
            @article{mandal2026grading,
              author    = {Mandal, D.},
              title     = {Automated Relative Grading With Dual Statistical Protocols:
                           A Transparent, Rule-Based Framework for
                           Higher-Education Assessment},
              journal   = {IEEE Transactions on Education},
              year      = {2026},
              volume    = {},
              number    = {},
              pages     = {},
              doi       = {10.1109/TE.2026.XXXXXXX},
              note      = {Software: https://github.com/dmandal/strict-university-grading}
            }""")
        st.code(bibtex, language="bibtex")

    with st.expander("📦 Dependencies"):
        st.markdown("""
| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.35 | Web framework |
| `pandas` | ≥ 2.1 | Data manipulation |
| `numpy` | ≥ 1.26 | Numerical computation |
| `altair` | ≥ 5.2 | Interactive visualisation |
        """)

    st.markdown("---")
    st.caption(
        "MIT License · © 2026 D. Mandal · "
        "[GitHub](https://github.com/dmandal/strict-university-grading) · "
        "Issues and pull requests welcome."
    )
