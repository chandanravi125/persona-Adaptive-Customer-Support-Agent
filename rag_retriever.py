import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# BM25
from langchain_community.retrievers import BM25Retriever

# Hybrid Search
from langchain.retrievers import EnsembleRetriever

# Multi Query
from langchain.retrievers.multi_query import MultiQueryRetriever

# Context Compression
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Set Google API key
google_API_key = os.getenv("GOOGLE_API_KEY")

def create_vector_db():
    """
    Creates a vector database using the provided text file.

    Returns:
        vectordb (Chroma): The created vector database.
    """
    try:
        loader = TextLoader("data/knowledge_base.txt", autodetect_encoding=True)
        documents = loader.load()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=google_API_key)
        vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory="chroma_db")
        vectordb.persist()
        return vectordb
    except Exception as e:
        print(f"Error: {e}")

def get_relevant_docs(query):
    """
    Retrieves relevant documents from the vector database based on the query.

    Args:
        query (str): The query to search for.

    Returns:
        results (list): A list of relevant documents.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=google_API_key)
        vectordb = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
        results = vectordb.similarity_search(query, k=2)
        return results
    except Exception as e:
        print(f"Error: {e}")



# def create_vector_db():

#     loader = TextLoader(
#         "data/knowledge_base.txt",
#         autodetect_encoding=True
#     )

#     documents = loader.load()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100
#     )

#     chunks = splitter.split_documents(documents)

#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=google_API_key
#     )

#     vectordb = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory="chroma_db"
#     )

#     vectordb.persist()

#     return vectordb

# def semantic_search(query):

#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=google_API_key
#     )

#     vectordb = Chroma(
#         persist_directory="chroma_db",
#         embedding_function=embeddings
#     )

#     return vectordb.similarity_search(
#         query,
#         k=5
#     )

# def mmr_search(query):

#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=google_API_key
#     )

#     vectordb = Chroma(
#         persist_directory="chroma_db",
#         embedding_function=embeddings
#     )

#     return vectordb.max_marginal_relevance_search(
#         query,
#         k=5,
#         fetch_k=20
#     )

# def bm25_search(query):

#     loader = TextLoader(
#         "data/knowledge_base.txt",
#         autodetect_encoding=True
#     )

#     docs = loader.load()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100
#     )

#     chunks = splitter.split_documents(docs)

#     bm25 = BM25Retriever.from_documents(chunks)

#     bm25.k = 5

#     return bm25.invoke(query)


# def hybrid_search(query):

#     loader = TextLoader(
#         "data/knowledge_base.txt",
#         autodetect_encoding=True
#     )

#     docs = loader.load()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100
#     )

#     chunks = splitter.split_documents(docs)

#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=google_API_key
#     )

#     vectordb = Chroma(
#         persist_directory="chroma_db",
#         embedding_function=embeddings
#     )

#     vector_retriever = vectordb.as_retriever(
#         search_kwargs={"k":5}
#     )

#     bm25 = BM25Retriever.from_documents(chunks)

#     bm25.k = 5

#     hybrid = EnsembleRetriever(
#         retrievers=[
#             bm25,
#             vector_retriever
#         ],
#         weights=[
#             0.4,
#             0.6
#         ]
#     )

#     return hybrid.invoke(query)

# def multi_query_search(query):

#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=google_API_key
#     )

#     vectordb = Chroma(
#         persist_directory="chroma_db",
#         embedding_function=embeddings
#     )

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         google_api_key=google_API_key
#     )

#     retriever = MultiQueryRetriever.from_llm(
#         retriever=vectordb.as_retriever(),
#         llm=llm
#     )

#     return retriever.invoke(query)

# def compressed_search(query):

#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         google_api_key=google_API_key
#     )

#     vectordb = Chroma(
#         persist_directory="chroma_db",
#         embedding_function=embeddings
#     )

#     retriever = vectordb.as_retriever()

#     compressor = EmbeddingsFilter(
#         embeddings=embeddings,
#         similarity_threshold=0.70
#     )

#     compression = ContextualCompressionRetriever(
#         base_retriever=retriever,
#         base_compressor=compressor
#     )

#     return compression.invoke(query)

# from sentence_transformers import CrossEncoder

# reranker = CrossEncoder("BAAI/bge-reranker-base")

# def rerank(query, docs):

#     pairs = [[query, doc.page_content] for doc in docs]

#     scores = reranker.predict(pairs)

#     ranked = sorted(
#         zip(scores, docs),
#         reverse=True,
#         key=lambda x: x[0]
#     )

#     return [doc for _, doc in ranked]

