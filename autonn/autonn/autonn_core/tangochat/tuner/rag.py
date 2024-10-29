import sys
from pathlib import Path
# working directory = /source/autonn_core/tangochat
CORE_DIR        = Path(__file__).resolve().parent.parent # /source/autonn_core
sys.path.append(str(CORE_DIR))
COMMON_ROOT     = Path("/shared/common")
DATASET_ROOT    = Path("/shared/datasets")
MODEL_ROOT      = Path("/shared/models")
CFG_DIR         = CORE_DIR / 'tangochat' / 'common' / 'cfg'

import logging
logger = logging.getLogger(__name__)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

def load_and_retrieve_docs(url, emb_model):
    '''
    WebBaseLoader.load(): 주어진 URL에서 웹 페이지를 로드하고, 이를 문서로 변환
    RecursiveCharacterTextSplitter.split_document(): 문서를 적절한 크기로 분할
    OllamaEmbeddings(): 분할된 문장들에 대해 각각의 문장에 대한 의미론적 임베딩 생성
    Chroma.from_documents(): 각 문장의 임베딩을 저장, 문맥 이해와 적합한 문장 검색
    '''
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict() 
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=emb_model) #"mxbai-embed-large"
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(url, question):
    retriever = load_and_retrieve_docs(url)
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    return formatted_prompt

def get_rag_formatted_prompt(retriever, question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    formatted_prompt = f"Question: {question}\n\nContext: {formatted_context}"
    return formatted_prompt