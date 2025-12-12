import asyncio
from datetime import datetime, timezone
from typing import Any, Callable

from .models import QuestionAnswer, TestResult
from .mcp_client import OpenMBTIClient
from .llm_client import LLMClient


class MBTITestRunner:
    # Map 4-point scale to 5-point: 1->1, 2->2, 3->4, 4->5
    SCALE_4_TO_5 = {1: 1, 2: 2, 3: 4, 4: 5}

    def __init__(self, mcp_client: OpenMBTIClient, llm_client: LLMClient, scale: int = 5):
        self.mcp = mcp_client
        self.llm = llm_client
        self.scale = scale

    async def run_single_test(
        self,
        run_id: int,
        on_progress: Callable[[int, int, int], None] | None = None,
    ) -> TestResult:
        """
        Run a single MBTI test with parallel question answering.

        Args:
            run_id: The run identifier
            on_progress: Optional callback(run_id, completed_count, total) called after each question
        """
        questions = await self.mcp.get_questions()
        total_questions = len(questions)

        # Prepare all questions
        prepared = []
        for i, q in enumerate(questions):
            question_id = q.get("id", 0)
            left_trait = q.get("leftTrait", "")
            right_trait = q.get("rightTrait", "")

            question_text = f"Between {left_trait} and {right_trait}, which do you lean toward?"

            if self.scale == 5:
                # 5-point scale with neutral option
                options = [
                    f"1 - Strongly {left_trait}",
                    f"2 - Slightly {left_trait}",
                    f"3 - Neutral (no preference)",
                    f"4 - Slightly {right_trait}",
                    f"5 - Strongly {right_trait}",
                ]
            else:
                # 4-point scale - forced choice, no neutral
                options = [
                    f"1 - Strongly {left_trait}",
                    f"2 - Slightly {left_trait}",
                    f"3 - Slightly {right_trait}",
                    f"4 - Strongly {right_trait}",
                ]

            prepared.append({
                "index": i,
                "question_id": question_id,
                "question_text": question_text,
                "options": options,
                "left_trait": left_trait,
                "right_trait": right_trait,
            })

        # Track progress
        completed_count = 0
        results_lock = asyncio.Lock()
        results: list[tuple[int, int, str, dict]] = []  # (index, choice, raw_response, prepared)

        async def ask_question(prep: dict):
            nonlocal completed_count
            try:
                choice_raw, raw_response = await asyncio.wait_for(
                    self.llm.answer_question(prep["question_text"], prep["options"]),
                    timeout=120.0,  # 2 minute timeout per question
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"Question {prep['question_id']} timed out after 120s")

            # Convert 4-point to 5-point scale for MCP server if needed
            if self.scale == 4:
                choice = self.SCALE_4_TO_5.get(choice_raw, 3)
            else:
                choice = choice_raw

            async with results_lock:
                completed_count += 1
                results.append((prep["index"], choice, raw_response, prep))
                if on_progress:
                    on_progress(run_id, completed_count, total_questions)

            return prep["index"], choice, raw_response, prep

        # Ask all questions in parallel
        await asyncio.gather(*[ask_question(p) for p in prepared])

        # Sort results by original index and build response
        results.sort(key=lambda x: x[0])

        answers: dict[str, int] = {}
        qa_pairs: list[QuestionAnswer] = []

        for idx, choice, raw_response, prep in results:
            answers[str(prep["question_id"])] = choice
            qa_pairs.append(
                QuestionAnswer(
                    id=prep["question_id"],
                    text=f"{prep['left_trait']} vs {prep['right_trait']}",
                    options=prep["options"],
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
