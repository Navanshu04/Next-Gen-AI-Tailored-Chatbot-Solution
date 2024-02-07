import os

import openai
from llama_index.llms import OpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]


class LoadLLM:
    def __init__(self):
        pass

    def load_llm(self):
        llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo", streaming=True)
        return llm
