import chainlit as cl
from llama_index import (
    ServiceContext,
)
from llama_index.callbacks import CallbackManager


class LoadServiceContext:
    def __init__(self, llm):
        self.llm = llm

    def load_service_context(self, llm):
        chunk_size = 1024

        service_context = ServiceContext.from_defaults(chunk_size=chunk_size, llm=llm,
                                                       callback_manager=CallbackManager(
                                                           [cl.LlamaIndexCallbackHandler()]), )
        return service_context
