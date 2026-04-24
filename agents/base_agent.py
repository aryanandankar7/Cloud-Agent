import os
import requests


class BaseAgent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("❌ GROQ_API_KEY not set")

    def _chat(self, messages):
        prompt = ""
        for m in messages:
            prompt += f"{m['role']}: {m['content']}\n"

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
            },
        )

        data = response.json()
        print("GROQ RESPONSE:", data)

        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        else:
            return f"❌ Groq API Error: {data}"

    def run(self, user_input):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]
        return self._chat(messages)