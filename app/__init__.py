"""StrictUniversityGrading — application package."""
from app.grading_engine import (
    GradingConfig,
    GradingEngine,
    GradingResult,
    CourseType,
    Protocol,
    GradingMethod,
    run_grading,
)

__all__ = [
    "GradingConfig", "GradingEngine", "GradingResult",
    "CourseType", "Protocol", "GradingMethod", "run_grading",
]
