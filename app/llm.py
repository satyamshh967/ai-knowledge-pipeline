import os

from dotenv import load_dotenv
from openai import OpenAI


class LLMService:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured"
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    def generate(
        self,
        question: str,
        context: str
    ) -> str:

        prompt = f"""
You are a knowledge assistant.

Answer the question using ONLY the provided context.

If the context does not contain enough information
to answer the question, say:

"I don't have enough information in the provided knowledge."

Do not invent facts.

CONTEXT:
{context}

QUESTION:
{question}
"""

        response = self.client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content