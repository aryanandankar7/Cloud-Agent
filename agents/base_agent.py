import os
import google.generativeai as genai


class BaseAgent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("❌ GEMINI_API_KEY not set")

        genai.configure(api_key=api_key)
       self.model = genai.GenerativeModel("gemini-1.5-flash-8b")

    def _chat(self, messages):
        prompt = ""

        for m in messages:
            prompt += f"{m['role']}: {m['content']}\n"

        response = self.model.generate_content(prompt)

        return response.text

    def run(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        return self._chat(messages)