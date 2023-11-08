from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext, load_index_from_storage,
)
from llama_index.node_parser import SimpleNodeParser

from utils import get_documents, get_llm, get_embed_model


class Indexing:
    def __init__(self):
        try:
            # rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            self.index = load_index_from_storage(storage_context)

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
            self.index = VectorStoreIndex.from_documents(
                documents, service_context=service_context, show_progress=True
            )
            self.index.storage_context.persist()
