from llama_index.query_engine import CustomQueryEngine
from llama_index.response_synthesizers import BaseSynthesizer
from llama_index.retrievers import BaseRetriever


class UnicampQueryEngine(CustomQueryEngine):
    """RAG Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)

        return response_obj

    async def acustom_query(self, query_str: str):
        """Run a custom query asynchronously."""
        # by default, just run the synchronous version
        return self.custom_query(query_str)
