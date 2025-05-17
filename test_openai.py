import os
from llama_index.llms.openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
    model=os.getenv("OPENAI_MODEL")
)

response = client.complete(
    prompt="What is the capital of France?"
)

print(response.text)
