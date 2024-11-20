from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_paths):
    pages = []
    for pdf_path in pdf_paths:  # Iterate over the list of file paths
        loader = PyPDFLoader(pdf_path)
        pages.extend(loader.load_and_split())  # Append pages from each document
    return pages  # Return all pages from all PDFs

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = splitter.split_documents(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OllamaEmbeddings(model="llama3.2")
    
    try:
        # Connect to Chroma with a persistent directory
        vector_store = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory="./chroma_db",  # Use persistent directory
        )
        
        print("Vector store successfully created.")
    
    except ValueError as e:
        print(f"Error creating vector store: {e}")
        raise
    
    return vector_store




def get_conversational_chain(vector_store):
    print("\n\n\nstart convo\n\n\n\n")
    llm = OllamaLLM(model="llama3.2")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return conversation_chain
