import os
import asyncio
import chainlit as cl

from llama_index.llms import openai
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Define a tool for multiplying two numbers
def multiply(a: float, b: float) -> float:
    return a * b

# Create an agent workflow with tools
agent = FunctionAgent(
    tools = [],
    llm = OpenAI(model="gpt-4o-mini"),
    system_prompt = "You are a helpful assistant that can multiply two numbers.",
)

async def main():
    while True:
        msg = input('How can I help you? Type E to exit: ')

        if msg == 'E' or msg == 'e':
            exit(0)
        
        response = await agent.run(user_msg = msg)
        print(str(response))

if __name__ == '__main__':
    asyncio.run(main())