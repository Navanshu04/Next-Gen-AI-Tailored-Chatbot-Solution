import os

import openai
from trulens_eval import Feedback
from trulens_eval import Tru
from trulens_eval import TruLlama
from trulens_eval.app import App
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI

openai.api_key = os.environ["OPENAI_API_KEY"]
tru = Tru()
openai = OpenAI()


class EvaluationLlm:

    def __init__(self, query_engine):
        self.query_engine = query_engine

    def evaluate(self, query_engine):
        context = App.select_context(query_engine)
        return context

    def groundness(self, context):
        grounded = Groundedness(groundedness_provider=OpenAI())
        f_groundedness = (
            Feedback(grounded.groundedness_measure_with_cot_reasons)
            .on(context.collect())  # collect context chunks into a list
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
        )
        return f_groundedness

    def relavance(self):
        f_qa_relevance = Feedback(openai.relevance).on_input_output()
        return f_qa_relevance

    def logging_app(self, query_engine, f_groundedness, f_qa_relevance):
        tru_query_engine_recorder = TruLlama(query_engine,
                                             app_id='chatbotpoc',
                                             feedbacks=[f_groundedness, f_qa_relevance])

        return tru_query_engine_recorder

    def evaluation_llm(self):
        context = self.evaluate(self.query_engine)
        f_groundedness = self.groundness(context)
        f_qa_relevance = self.relavance()
        tru_query_engine_recorder = self.logging_app(self.query_engine, f_groundedness, f_qa_relevance)
        return tru_query_engine_recorder
