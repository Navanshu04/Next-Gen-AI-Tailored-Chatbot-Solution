import os

import chainlit as cl
import openai
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.schema import (
    HumanMessage
)
from langchain_community.chat_models import ChatOpenAI
from llama_index import (
    ServiceContext,
    StorageContext,
)
from llama_index import (
    VectorStoreIndex,
    SQLDatabase,
)
from llama_index.callbacks import CallbackManager
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.indices.vector_store import VectorIndexAutoRetriever
from llama_index.llms import OpenAI
from llama_index.query_engine import SQLAutoVectorQueryEngine, RetrieverQueryEngine
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from pinecone import Pinecone
from pinecone import ServerlessSpec

from sqlalchemy import create_engine
os.environ["PINECONE_API_KEY"] = "YOUR KEY HERE"
api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)
# connect to the index
pinecone_index = pc.Index('chatbotpoc')

spec = ServerlessSpec(
    cloud="aws", region="us-west-2"
)
class VectorEngine:
    def __init__(self, service_context):
        self.service_context = service_context


    def load_vector(self):
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex([], storage_context=storage_context)
        return vector_index

    def vector_engine(self, vector_index, service_context):
        vector_store_info = VectorStoreInfo(
            content_info="articles about llama 2",
            metadata_info=[
                MetadataInfo(name="title", type="str", description="data about Llama 2"),
            ],
        )
        vector_auto_retriever = VectorIndexAutoRetriever(
            vector_index, vector_store_info=vector_store_info
        )
        retriever_query_engine = RetrieverQueryEngine.from_args(
            vector_auto_retriever, service_context=service_context
        )
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=retriever_query_engine,
            description=f"Useful for answering semantic questions about llama 2, which is a LLM developed by Meta"
        )
        return vector_tool

