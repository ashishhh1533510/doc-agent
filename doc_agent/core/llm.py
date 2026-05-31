"""
Model layer: the single place that connects to Gemini.

Everything that needs the LLM goes through build_agent(). Isolating the model
config here means we can swap models or providers without touching any tool,
agent, or workflow code.
"""

import os
from dotenv import load_dotenv
from agent_framework.openai import OpenAIChatCompletionClient

load_dotenv()


def build_agent(instructions: str, name: str):
    """Build a Gemini-backed agent with the given instructions and name."""
    return OpenAIChatCompletionClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.environ["GEMINI_API_KEY"],
        model="gemini-2.5-flash",
    ).as_agent(name=name, instructions=instructions)