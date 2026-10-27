import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found. Please set your environment variable.")

# Initialize client pointed at OpenRouter's endpoint
client = OpenAI(
    base_url="https://openrouter.ai/api/v1/",
    api_key=OPENROUTER_API_KEY,
)

# You can use any model supported by your OpenRouter account (e.g., Llama 3, Qwen, GPT-4o, etc.)
output = client.chat.completions.create(
    model="moonshotai/kimi-k3",
    messages=[
        {"role": "user", "content": "The capital of France is"},
    ],
    max_tokens=1024,
)

print(output.choices[0].message.content)