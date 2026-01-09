import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = None

def get_client():
    global _client

    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        _client = OpenAI(api_key=api_key)

    return _client


def call_gpt(messages, model="gpt-4o-mini"):
    client = get_client()
    
    # Auto-wrap string prompts
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
        
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content
