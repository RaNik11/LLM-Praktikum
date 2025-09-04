import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from api import openai_api
from bs4 import SoupStrainer

#Форматирую URL
bs4_strainer = SoupStrainer(class_=["post-title", "post-header", "post-content"])

loader = WebBaseLoader(
    web_path="",
    bs_kwargs={"parse_only": bs4_strainer},
)
#сохраняю в docs
docs = loader.load()
#проверка
print(f"Total characters: {len(docs[0].page_content)}")

#Разделяем дату на чанки
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap =200,
    add_start_index=True
)
all_splits = text_spliter.split_documents(docs)
print(f"Total splits: {len(all_splits)}")

#Чанки в вектора
embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small",
    openai_api_key =openai_api
)
#Изи сохраняю вектора в базу
vector_store = Chroma(
    collection_name="promt-engineering",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

ids = vector_store.add_documents(all_splits)

print(f"Persisted {len(ids)} documment to disk")
