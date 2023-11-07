from llama_index.prompts import PromptTemplate

synthesizer_text_qa_template = PromptTemplate(
    "As informações dos documentos estão apresentadas abaixo.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dadas as informações dos documentos e nenhum conhecimento prévio, "
    "responda a seguinte pergunta.\n"
    "Pergunta: {query_str}\n"
    "Resposta: "
)

synthesizer_refine_template = PromptTemplate(
    "A pergunta original é a seguinte: {query_str}\n"
    "Fornecemos uma resposta existente: {existing_answer}\n"
    "Temos a oportunidade de refinar a resposta existente (somente se necessário) com mais contexto abaixo.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Dado o novo contexto, refine a resposta original para melhor responder à pergunta. "
    "Se o contexto não for útil, retorne APENAS a resposta original.\n"
    "Resposta: "
)
