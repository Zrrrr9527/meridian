import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

def call_llm(model: str, messages: list[dict], temperature: float = 0):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        n=1,
        temperature=temperature,
    )

    return response.choices[0].message.content, (
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )
