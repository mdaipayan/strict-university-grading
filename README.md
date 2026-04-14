# StrictUniversityGrading

[![CI](https://github.com/dmandal/strict-university-grading/actions/workflows/ci.yml/badge.svg)](https://github.com/dmandal/strict-university-grading/actions)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![IEEE](https://img.shields.io/badge/Published-IEEE%20Trans.%20Educ.-00629B)](https://doi.org/10.1109/TE.2026.XXXXXXX)

**Companion software** for:

> D. Mandal, *"Automated Relative Grading With Dual Statistical Protocols: A Transparent, Rule-Based Framework for Higher-Education Assessment,"* **IEEE Transactions on Education**, 2026. DOI: [10.1109/TE.2026.XXXXXXX](https://doi.org/10.1109/TE.2026.XXXXXXX)

---

## Overview

`StrictUniversityGrading` automates the full grade-assignment pipeline for university courses, combining:

- **Pre-grading eligibility gates** (attendance, ESE absenteeism, ESE minimum marks)
- **Dual statistical protocols** for reference population selection (see §IV of the paper)
- **Safeguard conditions** that prevent pathological grade distributions
- **Interactive web dashboard** built with Streamlit

The system selects between *relative* (norm-referenced) and *absolute* (criterion-referenced) grading depending on cohort size (threshold *n* = 30), and exposes a full audit log for every grading decision.

---

## Five-Stage Pipeline

```
Input CSV
  │
  ├─[Stage 1]── Attendance < 75%  ──────────────────────► Grade = I
  │
  ├─[Stage 2]── ESE = "AB"        ──────────────────────► Grade = Z
  │
  ├─[Stage 3]── ESE score < 20% of ESE max ─────────────► Grade = F
  │
  ├─[Stage 4]── Build reference population (Protocol A or B)
  │             │
  │             ├── n* ≥ 30 → Relative grading (X̄ ± kσ)
  │             └── n* < 30 → Absolute grading (fixed thresholds)
  │
  └─[Stage 5]── Assign letter grade via boundary lookup
```

### Grade Boundaries (Relative Grading)

| Grade | Lower Boundary | Subject To |
|-------|---------------|------------|
| A+    | X̄ + 1.5 σ   | Cap at *M* |
| A     | X̄ + 1.0 σ   | — |
| B+    | X̄ + 0.5 σ   | — |
| B     | X̄            | — |
| C+    | X̄ − 0.5 σ  | Moderation rule |
| C     | X̄ − 1.0 σ  | Moderation rule |
| D     | X̄ − 1.5 σ  | Safeguards 1 & 2 |

### Safeguard Conditions

| # | Condition | Action |
|---|-----------|--------|
| 1 | D_raw > *P* (pass mark) | Moderation: collapse lower spine to *P* |
| 2 | D_raw < 0.30 *M* | Floor protection: shift entire curve up |

---

## Dual Statistical Protocols

| | Protocol A (Exclusive) | Protocol B (Inclusive) |
|-|------------------------|------------------------|
| Reference population | Students who passed ESE **and** attended | All attending students |
| Effect on X̄ | Higher (failures excluded) | Lower (zero-inflation) |
| Best for | Summative / high-stakes | Remedial / formative |
| Grade inflation | Controlled | Possible |

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/dmandal/strict-university-grading.git
cd strict-university-grading
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the web app

```bash
streamlit run main.py
```

The app opens at `http://localhost:8501`.

---

## Input Format

Upload a CSV with the following four columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | any | Student identifier |
| `marks` | numeric | Total course marks |
| `attendance` | numeric | Attendance percentage (0–100) |
| `ese_marks` | numeric **or** `"AB"` | ESE score; `"AB"` denotes absent |

A sample file is provided at `data/sample_cohort.csv`.

---

## Python API

The grading engine can be used independently of Streamlit:

```python
import pandas as pd
from app.grading_engine import GradingEngine, GradingConfig, CourseType, Protocol

df = pd.read_csv("data/sample_cohort.csv")

cfg = GradingConfig(
    total_max_marks = 100,
    ese_max_marks   = 60,
    course_type     = CourseType.THEORY,
    protocol        = Protocol.EXCLUSIVE,
)

result = GradingEngine(cfg).process(df)

print(f"Pass rate : {result.pass_rate:.1f}%")
print(f"Method    : {result.method.value}")
print(result.results_df[["id", "marks", "Final_Grade"]].head(10))
```

### Simulation

```python
from app.simulation import generate_cohort

df = generate_cohort(
    n            = 200,
    distribution = "bimodal",
    ese_fail_rate= 0.15,
    seed         = 42,
)
```

---

## Project Structure

```
strict-university-grading/
├── main.py                        # Streamlit dashboard (entry point)
├── requirements.txt
├── CITATION.cff                   # Machine-readable citation
├── LICENSE
├── README.md
│
├── app/
│   ├── __init__.py
│   ├── grading_engine.py          # Core pipeline (GradingEngine, GradingConfig)
│   ├── simulation.py              # Synthetic cohort generators
│   └── visualizations.py         # Altair chart builders
│
├── tests/
│   ├── __init__.py
│   └── test_grading_engine.py     # 30+ unit & integration tests
│
├── data/
│   └── sample_cohort.csv          # 50-student example
│
└── .github/
    └── workflows/
        └── ci.yml                 # GitHub Actions: test × lint × smoke
```

---

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=app
```

The test suite covers:
- All five pipeline stages
- Both protocols
- Both safeguard triggers
- Absolute vs relative modality switch
- Edge cases (empty cohort, single student, all-absent)

---

## Deploying to Streamlit Community Cloud

1. Fork this repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your fork, branch `main`, file `main.py`.
4. Click **Deploy**. No additional secrets required.

---

## Citing This Work

If you use this software or methodology, please cite the paper **and** the software:

```bibtex
@article{mandal2026grading,
  author  = {Mandal, D.},
  title   = {Automated Relative Grading With Dual Statistical Protocols:
             A Transparent, Rule-Based Framework for Higher-Education Assessment},
  journal = {IEEE Transactions on Education},
  year    = {2026},
  doi     = {10.1109/TE.2026.XXXXXXX},
  note    = {Software: https://github.com/dmandal/strict-university-grading}
}

@software{mandal2026software,
  author  = {Mandal, D.},
  title   = {StrictUniversityGrading},
  year    = {2026},
  url     = {https://github.com/dmandal/strict-university-grading},
  version = {1.0.0}
}
```

---

## Contributing

Pull requests are welcome. Please:

1. Fork the repo and create a feature branch (`git checkout -b feature/my-change`).
2. Add or update tests for any changed behaviour.
3. Ensure `pytest tests/ -v` passes locally.
4. Open a PR with a clear description of your changes.

---

## License

[MIT](LICENSE) © 2026 D. Mandal
