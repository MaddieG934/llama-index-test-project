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
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
from numpy import multiply

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load stored context
try:
    storage_context = StorageContext.from_defaults(persist_dir='./storage')
    index = load_index_from_storage(storage_context)
except:
    documents = SimpleDirectoryReader('./data').load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()

# Get tools
multiply_tool = FunctionTool.from_defaults(fn=multiply)

query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)
query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name='document search',
    description='searches provided documents'
)

# Start chat session
@cl.on_chat_start
async def start():

    # Create an agent workflow with tools
    agent = FunctionAgent(
        tools = [multiply_tool, query_tool],
        llm = OpenAI(model='gpt-4o-mini', temperature=0.1, max_tokens=1024, streaming=True),
        system_prompt = 'You are a helpful assistant that can multiply two numbers.',
    )

    # Set model, embeddings, context window, and callback manager
    Settings.llm = agent.llm
    Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')
    Settings.context_window = 4096
    Settings.callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()])

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