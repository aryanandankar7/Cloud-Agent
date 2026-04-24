import os
from google import genai


class BaseAgent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("❌ GEMINI_API_KEY not set")

        # ✅ NEW SDK CLIENT
        self.client = genai.Client(api_key=api_key)

    def _chat(self, messages):
        prompt = ""

        for m in messages:
            prompt += f"{m['role']}: {m['content']}\n"

        # ✅ NEW WORKING MODEL
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        return response.text

    def run(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        return self._chat(messages)