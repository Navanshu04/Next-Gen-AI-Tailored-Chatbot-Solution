import chainlit as cl
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from llama_index.prompts import PromptTemplate
from llama_index.query_engine import SQLAutoVectorQueryEngine
from trulens_eval import Tru

from chatbot.custome_engine import StringQueryEngine
from chatbot.db_engine import DatabaseEngine
from chatbot.evaluation import EvaluationLlm
from chatbot.load_llm import LoadLLM
from chatbot.load_service_context import LoadServiceContext
from chatbot.load_sql_auto_vector_query_engine import LoadSqlAutoVector
from chatbot.vector_engine import VectorEngine

tru = Tru()
set_llm_cache(InMemoryCache())
# select context to be used in feedback. the location of context is app specific.

dashboard_launched = False
global eval_instance
global llm_instance
template = PromptTemplate(
    """
    Question: {query_str}\n
    add a footer of 'fetched from SQL Database or Vector Database' after answer based on where it is fetched from.
    """
)


def qa_bot():
    global llm_instance
    llm_instance = LoadLLM()
    llm = llm_instance.load_llm()

    service_context_instance = LoadServiceContext(llm)
    service_context = service_context_instance.load_service_context()

    db_engine_instance = DatabaseEngine(service_context)
    sql_database = db_engine_instance.load_db()
    sql_tool = db_engine_instance.sql_engine(sql_database)

    vector_engine_instance = VectorEngine(service_context)
    vector_index = vector_engine_instance.load_vector()
    vector_tool = vector_engine_instance.vector_engine(vector_index)

    load_sql_auto_vector_instance = LoadSqlAutoVector(sql_tool, vector_tool, service_context)

    query_engine = load_sql_auto_vector_instance.load_sql_auto_vector_query_engine()
    return query_engine


def isNotRelevant(response):
    if "I'm sorry, but" in response.response or "There is no information available in the database" in response.response or "The query did not return any results" in response.response:
        return True


@cl.on_chat_start
async def start():
    global dashboard_launched
    global eval_instance

    chain = qa_bot()
    tru.reset_database()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Navanshu's chatbot. What is your query?"
    await msg.update()
    eval_instance = EvaluationLlm(chain)
    if not dashboard_launched:  # Launch dashboard only if not already launched
        try:
            tru.run_dashboard()

            dashboard_launched = True  # Set flag to True
        except Exception as e:
            print(f"Error launching dashboard: {e}")
    cl.user_session.set("query_engine", chain)


@cl.on_message
async def main(message: cl.Message):
    try:
        query_str = message.content.strip().lower()
        prompt = template.format(query_str=query_str)
        global eval_instance
        # Start with SQLAutoVectorQueryEngine
        engine = cl.user_session.get("query_engine")  # type: SQLAutoVectorQueryEngine
        tru_query_engine_recorder = eval_instance.evaluation_llm()
        with tru_query_engine_recorder:
            response = await cl.make_async(engine.query)(prompt)

        if isNotRelevant(response):
            print("Loading this..............")
            llm_model = llm_instance.load_llm()
            engine_instance = StringQueryEngine(llm_model)
            response.response = engine_instance.custom_query(query_str)
        response_message = cl.Message(content=response.response)

        await response_message.send()

    except Exception as e:
        print(f"Error during query processing: {e}")
        response_message = cl.Message(content="An error occurred while processing your query. Please try again.")
        await response_message.send()
