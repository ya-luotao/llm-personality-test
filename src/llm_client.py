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

        prompt = f"""You are taking an MBTI personality test. Answer the following question by choosing one option.

Question: {question}

Options:
{options_text}

Instructions:
- Choose the option that best describes you
- Reply with ONLY the number (1-{len(options)}) of your choice
- Do not explain your choice, just provide the number

Your choice:"""

        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are taking a personality test. Answer questions honestly based on how you would naturally respond. Only output the number of your choice.",
                        },
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
