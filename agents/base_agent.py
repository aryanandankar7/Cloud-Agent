import os
import google.generativeai as genai


class BaseAgent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt

        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("❌ GEMINI_API_KEY not set")

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Use working model
        self.model = genai.GenerativeModel("models/gemini-1.5-flash")

    def _chat(self, messages):
        prompt = ""

        # Convert messages into single prompt
        for m in messages:
            prompt += f"{m['role']}: {m['content']}\n"

        # Call Gemini
        response = self.model.generate_content(prompt)

        return response.text

    def run(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        return self._chat(messages)