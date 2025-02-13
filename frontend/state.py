# state.py
import os

from openai import AsyncOpenAI

import reflex as rx


class State(rx.State):
    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]]

    async def answer(self):
        # Our chatbot has some brains now!
        client = AsyncOpenAI(
            api_key="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"
        )

        session = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": self.question}
            ],
            stop=None,
            temperature=0.7,
            stream=True,
        )

        # Add to the answer as the chatbot responds.
        answer = ""
        self.chat_history.append((self.question, answer))

        # Clear the question input.
        self.question = ""
        # Yield here to clear the frontend input before continuing.
        yield

        async for item in session:
            if hasattr(item.choices[0].delta, "content"):
                if item.choices[0].delta.content is None:
                    # presence of 'None' indicates the end of the response
                    break
                answer += item.choices[0].delta.content
                self.chat_history[-1] = (
                    self.chat_history[-1][0],
                    answer,
                )
                yield
