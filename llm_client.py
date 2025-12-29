# ============================================================
# LLM CLIENT (USED BY PART 1+)
# ============================================================

import openai
import json
import time
from typing import List, Dict


class RobustLLMClient:
    """Rate-limited client with JSON enforcement and retries"""

    def __init__(self, base_url, api_key):
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.rate_limit_delay = 0.6  # 2 req/sec = 0.5s + buffer
        self.last_call = 0

    def _safe_json_call(self, model: str, messages: List[Dict], **kwargs) -> str:
        """Call with JSON mode + retries + rate limiting"""

        # Rate limiting
        now = time.time()
        time.sleep(max(0, self.rate_limit_delay - (now - self.last_call)))
        self.last_call = time.time()

        # JSON-only system prompt
        json_system = (
            "You are a JSON generator. ALWAYS respond with VALID JSON only.\n"
            "NEVER include markdown, explanations, or extra text. Just pure JSON.\n"
            'If you cannot generate valid JSON, return: {"error": "failed"}'
        )

        messages = [{"role": "system", "content": json_system}] + messages

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 800),
                )

                content = response.choices[0].message.content.strip()
                parsed = json.loads(content)  # validate JSON
                return json.dumps(parsed)

            except json.JSONDecodeError as e:
                print(f"JSON parse error (attempt {attempt+1}): {str(e)[:100]}")
                time.sleep(1)

            except Exception as e:
                print(f"API error (attempt {attempt+1}): {str(e)}")
                time.sleep(2)

        return json.dumps({"error": "failed", "attempts": 3})

    def chat(self, model: str, messages: List[Dict], **kwargs) -> str:
        """Public chat interface"""
        return self._safe_json_call(model, messages, **kwargs)
