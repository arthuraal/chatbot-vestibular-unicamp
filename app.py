import os

import chainlit as cl
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext, load_index_from_storage,
)
from llama_index.node_parser import SimpleNodeParser

from prompts import synthesizer_refine_template, synthesizer_text_qa_template
from query_engine import UnicampQueryEngine
from utils import get_documents, get_llm, get_embed_model, get_retriever

openai.api_key = os.environ.get("OPENAI_API_KEY")


def load_index():
    try:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)

    except:
        node_parser = SimpleNodeParser.from_defaults(
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=500)
        )

        llm = get_llm(llm_type="openai")
        embed_model = get_embed_model(embed_type="openai")

        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embed_model, node_parser=node_parser
        )

        documents = get_documents("https://www.pg.unicamp.br/norma/31594/0")

        # build index
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context, show_progress=True
        )
        index.storage_context.persist()

    return index


index = load_index()
retriever, response_synthesizer = get_retriever(index)

response_synthesizer.update_prompts(
    {"text_qa_template": synthesizer_text_qa_template, "refine_template": synthesizer_refine_template}
)


@cl.on_chat_start
async def factory():
    query_engine = UnicampQueryEngine(
        retriever=retriever, response_synthesizer=response_synthesizer
    )

    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")
    response = await cl.make_async(query_engine.custom_query)(message.content)

    response_message = cl.Message(content="")
    if response:
        response_message.content = str(response)

    await response_message.send()
