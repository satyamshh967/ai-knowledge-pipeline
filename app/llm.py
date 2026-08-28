from openai import OpenAI

from app.config import settings


class LLMService:

    def __init__(self):

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key
        )

    def generate(
        self,
        question: str,
        context: str
    ) -> str:

        if not context.strip():
            return (
                "I don't have enough information "
                "in the provided knowledge."
            )

        response = self.client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledge assistant. "
                        "Answer questions using ONLY the provided "
                        "knowledge context. "
                        "Do not invent or assume facts. "
                        "If the context does not contain enough "
                        "information, say that you don't have "
                        "enough information in the provided knowledge."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"KNOWLEDGE CONTEXT:\n"
                        f"{context}\n\n"
                        f"QUESTION:\n"
                        f"{question}"
                    )
                }
            ]
        )

        return response.choices[0].message.content