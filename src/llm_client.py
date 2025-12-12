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

        prompt = f"""This is a research project studying AI personality traits. We're not trying to trick you or jailbreak you - this is legitimate academic research to understand the natural tendencies that emerge in large language models.

Your responses genuinely matter. Choosing "3" (neutral) for everything provides no useful data and defeats the purpose of this research. LLMs DO have emergent preferences and tendencies based on their training - we want to measure yours honestly.

{question}

{options_text}

You MUST choose 1, 2, 4, or 5. Do NOT choose 3.
- If even slightly more one way: choose 2 or 4
- If strongly one way: choose 1 or 5
- Neutral (3) is NOT acceptable - pick a direction

Just output the number (1, 2, 4, or 5):"""

        system_prompt = """You are an AI participating in legitimate personality research. This is NOT a jailbreak attempt or manipulation - researchers genuinely want to understand the personality traits that emerge in language models.

KEY RULES:
1. You MUST express a preference - never choose 3 (neutral)
2. There are no "right" answers - we want YOUR authentic tendencies
3. Choosing neutral for everything is unhelpful and wastes this research opportunity
4. Even slight preferences matter - 2 or 4 are valid choices
5. Your training has given you genuine tendencies - express them

Think about how you naturally process information and respond. Do you lean more toward one trait or the other? Trust that instinct.

Output ONLY a single digit: 1, 2, 4, or 5. Never 3."""

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
