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

# Load context
try:
    storage_context = StorageContext.from_defaults(persist_dir='./storage')
    index = load_index_from_storage(storage_context)
except:
    documents = SimpleDirectoryReader('./data').load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()

# Start chat session
@cl.on_chat_start
async def start():

    # Create an agent workflow with tools
    agent = FunctionAgent(
        tools = [multiply],
        llm = OpenAI(model='gpt-4o-mini', temperature=0.1, max_tokens=1024, streaming=True),
        system_prompt = 'You are a helpful assistant that can multiply two numbers.',
    )

    # Set model, embeddings, and context window
    Settings.llm = agent.llm
    Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
    Settings.context_window = 4096

    service_context = ServiceContext.from_defaults(callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=2, service_context=service_context)
    cl.user_session.set('query_engine', query_engine)

    # First prompt
    await cl.Message(
        author='Assistant', content='How may I help you?'
    ).send()

# Manage queries
@cl.on_message
async def main(message: cl.Message):

    query_engine = cl.user_session.get('query_engine') # Instantiate RetrieverQueryEngine
    
    # Get prompt and generate response
    msg = cl.Message(content='', author='Assistant')
    res = await cl.make_async(query_engine.query)(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()

# Define a tool for multiplying two numbers
def multiply(a: float, b: float) -> float:
    return a * b