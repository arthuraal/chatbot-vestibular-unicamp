from llama_index import get_response_synthesizer
from llama_index.indices.vector_store import VectorIndexRetriever


class Querying:
    def __init__(self, index):
        self.index = index

    def build_retriever(self):
        # configure retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=4,
        )
        return retriever

    @staticmethod
    def build_synthesizer():
        # configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
        )

        return response_synthesizer
