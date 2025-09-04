from langchain.chains import llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from api import openai_api

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api
)

vector_store = Chroma(
    collection_name="promt-engineering",
    embedding_function=embeddings,
    persist_directory="./chroma_db", )

promt = ChatPromptTemplate.from_template(
    """Тут пишем промт настройка для нашего LLM Чат бота т.е то как он себя должен ввести как отвечать и тд
    Question: {question}
    Context: {context}
    Answer:"""
)

lmm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api)

question = "Вопросы"

retrieved_docs = vector_store.similarity_search(question, k=3)
docs_content ="\n".join(doc.page_content for doc in retrieved_docs)

message = promt.invoke({"question": question, "context": docs_content})

answer = llm.invoke(message)
print(answer.content)
