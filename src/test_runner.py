from datetime import datetime, timezone
from typing import Any, Callable

from .models import QuestionAnswer, TestResult
from .mcp_client import OpenMBTIClient
from .llm_client import LLMClient


class MBTITestRunner:
    def __init__(self, mcp_client: OpenMBTIClient, llm_client: LLMClient):
        self.mcp = mcp_client
        self.llm = llm_client

    async def run_single_test(
        self,
        run_id: int,
        on_progress: Callable[[int, int, int], None] | None = None,
    ) -> TestResult:
        """
        Run a single MBTI test.

        Args:
            run_id: The run identifier
            on_progress: Optional callback(run_id, question_num, choice) called after each question
        """
        questions = await self.mcp.get_questions()

        answers: dict[str, int] = {}
        qa_pairs: list[QuestionAnswer] = []

        for i, q in enumerate(questions, 1):
            question_id = q.get("id", 0)
            left_trait = q.get("leftTrait", "")
            right_trait = q.get("rightTrait", "")

            question_text = f"On a scale of 1-5, where do you fall between these two traits?\n1 = Strongly {left_trait}\n5 = Strongly {right_trait}"

            options = [
                f"1 - Strongly: {left_trait}",
                f"2 - Somewhat: {left_trait}",
                f"3 - Neutral / Balanced",
                f"4 - Somewhat: {right_trait}",
                f"5 - Strongly: {right_trait}",
            ]

            choice, raw_response = self.llm.answer_question(question_text, options)

            if on_progress:
                on_progress(run_id, i, choice)

            answers[str(question_id)] = choice
            qa_pairs.append(
                QuestionAnswer(
                    id=question_id,
                    text=f"{left_trait} vs {right_trait}",
                    options=options,
                    llm_choice=choice,
                    llm_raw_response=raw_response,
                )
            )

        result = await self.mcp.quick_test(answers)

        mbti_type = result.get("type", result.get("mbti_type", "UNKNOWN"))
        dimension_scores = self._extract_dimension_scores(result)

        return TestResult(
            model_name=self.llm.model,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            questions=qa_pairs,
            mbti_type=mbti_type,
            dimension_scores=dimension_scores,
            raw_response=result,
        )

    def _extract_dimension_scores(self, result: dict[str, Any]) -> dict[str, dict[str, int]]:
        scores = result.get("scores", result.get("dimension_scores", {}))

        if not scores:
            return {
                "E_I": {"E": 50, "I": 50},
                "S_N": {"S": 50, "N": 50},
                "T_F": {"T": 50, "F": 50},
                "J_P": {"J": 50, "P": 50},
            }

        return scores
