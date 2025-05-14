import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Cargar el .env
load_dotenv()

# Leer las claves desde el entorno
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")

# No necesitamos inicializar pinecone explícitamente
# Solo LangchainPinecone se encarga de eso

FILE_LIST = "archivos.txt"

def save_name_files(path, new_files):
    old_files = load_name_files(path)
    with open(path, "a") as file:
        for item in new_files:
            if item not in old_files:
                file.write(item + "\n")
                old_files.append(item)
    return old_files

def load_name_files(path):
    archivos = []
    if os.path.exists(path):
        with open(path, "r") as file:
            archivos = [line.strip() for line in file]
    return archivos

def clean_files(path):
    open(path, "w").close()
    # Ya no necesitamos hacer nada con pinecone, solo limpiamos el índice con LangchainPinecone
    index = LangchainPinecone.from_existing_index(INDEX_NAME)
    index.delete(delete_all=True)
    return True

def text_to_pinecone(pdf):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, pdf.name)
    with open(temp_filepath, "wb") as f:
        f.write(pdf.getvalue())

    loader = PyPDFLoader(temp_filepath)
    text = loader.load()

    with st.spinner(f'Creando embedding fichero: {pdf.name}'):
        create_embeddings(pdf.name, text)

    return True

def create_embeddings(file_name, text):
    print(f"Creando embeddings del archivo: {file_name}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_documents(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Crear el índice en Pinecone a través de LangchainPinecone
    LangchainPinecone.from_documents(
        chunks,
        embeddings,
        index_name=INDEX_NAME
    )
    
    return True
