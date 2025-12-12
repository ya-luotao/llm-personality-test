import asyncio
import re
from openai import AsyncOpenAI


class LLMClient:
    def __init__(self, model: str, api_key: str, timeout: float = 60.0, temperature: float = 0.7):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/ya-luotao/llm-personality-test",
                "X-Title": "llm-personality-test",
            },
            timeout=timeout,
        )
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    async def answer_question(
        self, question: str, options: list[str], max_retries: int = 3
    ) -> tuple[int, str, int, int]:
        options_text = "\n".join(
            f"{i + 1}. {opt}" for i, opt in enumerate(options)
        )

        num_options = len(options)
        prompt = f"""{question}

{options_text}

Pick one (1-{num_options}):"""

        system_prompt = f"""Answer each question with your honest preference. Just output the number (1-{num_options})."""

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=10240,  # High limit for very verbose models like Gemini
                )

                message = response.choices[0].message
                raw_response = message.content or ""

                # Check for refusal (OpenAI API may include refusal field)
                refusal = getattr(message, 'refusal', None)
                if refusal:
                    raise ValueError(f"Model refused to answer: {refusal}")

                # Check for empty response (common sign of content filtering)
                if not raw_response.strip():
                    finish_reason = response.choices[0].finish_reason
                    raise ValueError(f"Model returned empty response (finish_reason: {finish_reason})")

                choice = self._parse_choice(raw_response, len(options), question)

                # Extract token usage
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                return choice, raw_response, input_tokens, output_tokens
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    await asyncio.sleep(wait_time)

        raise last_error or Exception("Failed after retries")

    async def answer_all_questions(
        self, questions: list[dict], max_retries: int = 3
    ) -> tuple[list[int], str, int, int]:
        """Answer all questions in a single API call.

        Args:
            questions: List of dicts with 'id', 'text', 'options'

        Returns:
            (choices, raw_response, input_tokens, output_tokens)
        """
        num_options = len(questions[0]["options"]) if questions else 5

        # Build the prompt with all questions
        questions_text = []
        for i, q in enumerate(questions, 1):
            options_text = " | ".join(q["options"])
            questions_text.append(f"{i}. {q['text']}\n   Options: {options_text}")

        all_questions = "\n\n".join(questions_text)

        prompt = f"""Answer all {len(questions)} questions below. For each question, pick one option (1-{num_options}).

{all_questions}

Reply with ONLY the numbers, one per line (e.g., "3" or "1"), in order from question 1 to {len(questions)}:"""

        system_prompt = f"""Answer each question with your honest preference. Output {len(questions)} numbers (1-{num_options}), one per line, in order."""

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=1024,  # Enough for 32 single-digit answers + some text
                )

                message = response.choices[0].message
                raw_response = message.content or ""

                # Check for refusal
                refusal = getattr(message, 'refusal', None)
                if refusal:
                    raise ValueError(f"Model refused to answer: {refusal}")

                # Check for empty response
                if not raw_response.strip():
                    finish_reason = response.choices[0].finish_reason
                    raise ValueError(f"Model returned empty response (finish_reason: {finish_reason})")

                # Parse all choices
                choices = self._parse_all_choices(raw_response, len(questions), num_options)

                # Extract token usage
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                return choices, raw_response, input_tokens, output_tokens
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)

        raise last_error or Exception("Failed after retries")

    def _parse_all_choices(self, response: str, expected_count: int, num_options: int) -> list[int]:
        """Parse multiple choices from response."""
        import re

        # Find all numbers in the response
        numbers = re.findall(r'\b(\d+)\b', response)

        choices = []
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= num_options:
                choices.append(num)
                if len(choices) >= expected_count:
                    break

        if len(choices) < expected_count:
            raise ValueError(
                f"Could not parse {expected_count} valid choices (1-{num_options}) from response. "
                f"Found {len(choices)}: {choices}"
            )

        return choices

    def _parse_choice(self, response: str, num_options: int, question: str = "") -> int:
        response = response.strip()

        match = re.search(r"\d+", response)
        if match:
            choice = int(match.group())
            if 1 <= choice <= num_options:
                return choice

        # Fail if can't parse valid choice
        short_question = question[:50] + "..." if len(question) > 50 else question
        raise ValueError(
            f"Could not parse valid choice (1-{num_options}) from response '{response}' "
            f"for question: '{short_question}'"
        )
