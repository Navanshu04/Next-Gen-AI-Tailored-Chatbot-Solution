from llama_index import PromptTemplate


class StringQueryEngine():
    qa_prompt = PromptTemplate(

        "---------------------\n"
        "Given the Query information, Please fetch the results and add a footer of 'fetched from LLM' after answer,\n "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    def __init__(self, llm):
        self.llm = llm

    def custom_query(self, query_str: str):
        prompt = self.qa_prompt.format(query_str=query_str)
        response = self.llm.complete(prompt)
        print(response)
        return str(response)
