import os
from dotenv import load_dotenv
from agent_framework.openai import OpenAIChatCompletionClient

load_dotenv()

def build_agent():
    return OpenAIChatCompletionClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-2.5-flash",
    ).as_agent(
        name="DocAgent",
        instructions="You are a helpful assistant. Answer in one sentence.",
    )