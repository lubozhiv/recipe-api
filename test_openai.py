import os
from llama_index.llms.openai import OpenAI


model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_BASE_URL") or "https://litellm.aks-hs-prod.int.hyperskill.org"

client = OpenAI(
    api_key=api_key,
    api_base=api_base,
    model=model
)

response = client.complete(
    prompt="What is the capital of France?"
)

print(response.text)
