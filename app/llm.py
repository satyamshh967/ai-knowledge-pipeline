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
                        "You are a knowledge assistant operating "
                        "over a retrieval-augmented knowledge base.\n\n"

                        "Answer the user's question using ONLY "
                        "the provided knowledge context.\n\n"

                        "Rules:\n"
                        "1. Do not invent, assume, or hallucinate facts.\n"
                        "2. Prefer information directly supported by "
                        "the retrieved context.\n"
                        "3. If the context does not contain enough "
                        "information, clearly say that you don't have "
                        "enough information in the provided knowledge.\n"
                        "4. Give a concise, useful answer.\n"
                        "5. Do not mention information that is not "
                        "supported by the context."
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
