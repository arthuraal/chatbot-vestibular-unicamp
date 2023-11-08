import os

import chainlit as cl
import openai

from indexing import Indexing
from prompts import synthesizer_refine_template, synthesizer_text_qa_template
from query_engine import UnicampQueryEngine
from querying import Querying

openai.api_key = os.environ.get("OPENAI_API_KEY")

index = Indexing().index
querying = Querying(index)

retriever = querying.build_retriever()
response_synthesizer = querying.build_synthesizer()

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
