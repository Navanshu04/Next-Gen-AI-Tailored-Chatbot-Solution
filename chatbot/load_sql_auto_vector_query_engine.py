from llama_index.query_engine import SQLAutoVectorQueryEngine


class LoadSqlAutoVector:

    def __init__(self, sql_tool, vector_tool, service_context):
        self.sql_tool = sql_tool
        self.vector_tool = vector_tool
        self.service_context = service_context

    def load_sql_auto_vector_query_engine(self):
        query_engine_llm = SQLAutoVectorQueryEngine(
            self.sql_tool, self.vector_tool, service_context=self.service_context
        )
        return query_engine_llm
