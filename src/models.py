from dataclasses import dataclass, field, asdict
from typing import Any
from datetime import datetime


@dataclass
class QuestionAnswer:
    id: int
    text: str
    options: list[str]
    llm_choice: int
    llm_raw_response: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DimensionScore:
    first: int
    second: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TestResult:
    model_name: str
    run_id: int
    timestamp: str
    questions: list[QuestionAnswer]
    mbti_type: str
    dimension_scores: dict[str, dict[str, int]]
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "mbti_type": self.mbti_type,
            "dimension_scores": self.dimension_scores,
            "questions": [q.to_dict() for q in self.questions],
            "raw_response": self.raw_response,
        }


@dataclass
class TestSession:
    model: str
    total_runs: int
    timestamp: str
    runs: list[TestResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        mbti_counts: dict[str, int] = {}
        for run in self.runs:
            mbti_counts[run.mbti_type] = mbti_counts.get(run.mbti_type, 0) + 1

        return {
            "model": self.model,
            "total_runs": self.total_runs,
            "timestamp": self.timestamp,
            "runs": [r.to_dict() for r in self.runs],
            "summary": {
                "mbti_distribution": mbti_counts,
                "completed_runs": len(self.runs),
            },
        }
