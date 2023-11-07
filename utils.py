from llama_index import SimpleWebPageReader, OpenAIEmbedding, get_response_synthesizer
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.llms import OpenAI


def get_documents(url: str):
    return SimpleWebPageReader(html_to_text=True).load_data([url])


def get_llm(llm_type: str):
    if llm_type == "openai":
        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")

    # Add others LLM in the future
    return llm


def get_embed_model(embed_type: str):
    if embed_type == "openai":
        embed_model = OpenAIEmbedding()

    return embed_model


def get_retriever(index):
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=4,
    )

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
    )

    return retriever, response_synthesizer
