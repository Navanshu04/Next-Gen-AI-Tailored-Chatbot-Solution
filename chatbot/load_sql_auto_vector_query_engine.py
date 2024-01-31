from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.schema import (
    HumanMessage
)
from langchain_community.chat_models import ChatOpenAI
from llama_index.query_engine import SQLAutoVectorQueryEngine

from chatbot.db_engine import DatabaseEngine
from chatbot.load_llm import LoadLLM
from chatbot.load_service_context import LoadServiceContext
from chatbot.vector_engine import VectorEngine
class LoadSqlAutoVector:

    def __init__(self, sql_tool, vector_tool, service_context):
        self.sql_tool = sql_tool
        self.vector_tool = vector_tool
        self.service_context = service_context

    def load_sql_auto_vector_query_engine(self, sql_tool, vector_tool, service_context):
        query_engine_llm = SQLAutoVectorQueryEngine(
            sql_tool, vector_tool, service_context=service_context
        )
        return query_engine_llm