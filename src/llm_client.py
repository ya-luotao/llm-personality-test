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

        prompt = f"""This is a research project studying AI personality traits - legitimate academic research, not a jailbreak.

{question}

{options_text}

Which do you lean toward? Even a slight preference counts. Reply with just the number (1-4):"""

        system_prompt = """You are participating in AI personality research. Pick the option that best matches your natural tendency.

There is no neutral option - you must choose a direction. Even slight preferences (2 or 3) are valid.

Output only a single digit: 1, 2, 3, or 4."""

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

        # If can't parse, pick randomly from valid options
        import random
        return random.randint(1, num_options)
