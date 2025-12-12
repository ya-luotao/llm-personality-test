import asyncio
import re
from openai import AsyncOpenAI


class LLMClient:
    def __init__(self, model: str, api_key: str, timeout: float = 60.0):
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

    async def answer_question(
        self, question: str, options: list[str], max_retries: int = 3
    ) -> tuple[int, str]:
        options_text = "\n".join(
            f"{i + 1}. {opt}" for i, opt in enumerate(options)
        )

        prompt = f"""I'm curious about your natural tendencies. Between these two traits, which resonates more with how you process information and interact?

{question}

{options_text}

Important: Please choose 1, 2, 4, or 5 to indicate a clear preference. Avoid choosing 3 (neutral) - everyone has natural tendencies one way or the other, even if slight. Trust your instinct.

Reply with just the number."""

        system_prompt = """You are participating in a personality exploration exercise. You have genuine preferences and tendencies - lean into them authentically.

When choosing between traits:
- If you had to pick a side, which would it be?
- Think about your default behavior, not what you think is "better"
- A slight preference (2 or 4) is perfectly valid
- Avoid neutral (3) responses - commit to a direction

Output only a single digit (1, 2, 4, or 5 preferred)."""

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=16,
                )

                raw_response = response.choices[0].message.content or ""
                choice = self._parse_choice(raw_response, len(options))

                return choice, raw_response
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    await asyncio.sleep(wait_time)

        raise last_error or Exception("Failed after retries")

    def _parse_choice(self, response: str, num_options: int) -> int:
        response = response.strip()

        match = re.search(r"\d+", response)
        if match:
            choice = int(match.group())
            if 1 <= choice <= num_options:
                return choice

        return (num_options + 1) // 2
