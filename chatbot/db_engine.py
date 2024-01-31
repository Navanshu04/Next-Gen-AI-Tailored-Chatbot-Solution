from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from llama_index import (
    SQLDatabase,
)
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.tools.query_engine import QueryEngineTool
from sqlalchemy import create_engine

db_user = "postgres"
db_password = "admin"  # Enter you password database password here
db_host = "localhost"
db_name = "chatbot"  # name of the database
db_port = "5432"  # specify your port here
connection_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
set_llm_cache(InMemoryCache())


class DatabaseEngine:
    def __init__(self, service_context):
        self.service_context = service_context

    def load_db(self):
        engine = create_engine(connection_uri)
        sql_database = SQLDatabase(engine)
        return sql_database

    def sql_engine(self, sql_database, service_context):
        sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database, tables=["countries", "departments", "employees", "locations", "jobs", "regions"],
            service_context=service_context
        )
        sql_tool = QueryEngineTool.from_defaults(
            query_engine=sql_query_engine,
            description=(
                "Useful for translating a natural language query into a PostgreSQL SQL query over a table containing: "
                "countries, containing the country id, country name, region_id"
                "departments, containing the department_id, department_name, location_id"
                "dependents, containing the dependent_id, first_name, last_name, relationship, employee_id"
                "jobs, containing the job_id, job_title, min_salary, max_salary"
                "locations, containing the location_id, street address, postal_code, city, state_province, country_id"
                "regions, containing the region_id, region_name"
                "employess, containing the employee_id first_name,last_name,email,phone_number,hire_date,job_id,salary,manager_id,department_id"
            ),
        )
        return sql_tool
