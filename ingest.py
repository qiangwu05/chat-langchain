"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def ingest_docs():
    """Get documents from web pages."""
    loader = BSHTMLLoader("langchain.readthedocs.io/en/latest/content_1882895.htm")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore_law.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
