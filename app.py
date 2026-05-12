import os
import chainlit as cl

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
from llama_index.core.callbacks import CallbackManager

os.environ['OPENAI_API_KEY']

# Load stored context
try:
    storage_context = StorageContext.from_defaults(persist_dir='./storage')
    index = load_index_from_storage(storage_context)
except:
    documents = SimpleDirectoryReader('./data').load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir='./storage')

# Get tools

# Multiply two numbers tool
def multiply(a: float, b: float):
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

# Query tool
query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)

query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name='document search',
    description='searches provided documents'
)

# Create an agent workflow with tools
agent = FunctionAgent(
    tools = [multiply_tool, query_tool],
    llm = OpenAI(model='gpt-4o-mini', temperature=0.1, max_tokens=1024, streaming=True),
    system_prompt = 'You are a helpful assistant that can multiply two numbers and answer questions about provided documents.',
)

# Start chat session
@cl.on_chat_start
async def start():

    # Set embeddings, context window, and callback manager
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
    
    # Get prompt and generate response
    msg = cl.Message(content='', author='Assistant')
    res = await agent.stream_chat(message.content)

    async for token in res.async_response_gen:
        await msg.stream_token(token)
    await msg.send()