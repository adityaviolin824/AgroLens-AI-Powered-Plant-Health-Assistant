import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from utils.exception import CustomException
from utils.logger import logging
from utils.read_yaml import read_yaml

CONFIG_PATH = Path("configuration-files/vectorstore_config.yaml")

def load_env_from_config(cfg) -> None:
    dotenv_override = bool(cfg.env.dotenv_override)
    load_dotenv(override=dotenv_override)
    key_name = cfg.env.openai_key_env
    os.environ["OPENAI_API_KEY"] = os.getenv(key_name, "your-key-if-not-using-env")
    logging.info("Environment variables loaded.")

def load_documents(folder: str, glob_pattern: str) -> list[Document]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"TXT folder does not exist: {folder}")

    loader = DirectoryLoader(
        folder,
        glob=glob_pattern,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        use_multithreading=True,
        show_progress=True,
    )

    documents: list[Document] = []
    for doc in loader.load():
        parent = os.path.basename(os.path.dirname(doc.metadata.get("source", ""))) or os.path.basename(folder)
        doc.metadata["doc_type"] = parent
        documents.append(doc)

    if not documents:
        raise ValueError(f"No .txt documents found under: {folder}")
    logging.info(f"Loaded {len(documents)} documents from '{folder}'.")
    return documents

def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    logging.info(f"Chunked into {len(chunks)} chunks.")
    return chunks

def build_embeddings(embedding_model: str) -> OpenAIEmbeddings:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    _ = embeddings.embed_query("ping")
    logging.info(f"Embeddings initialized with model '{embedding_model}'.")
    return embeddings

def create_and_persist_vectorstore(
    chunks: list[Document], 
    embeddings: OpenAIEmbeddings, 
    collection_name: str,
    persist_directory: str
) -> Chroma:
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    logging.info(f"Vectorstore created and persisted to: {persist_directory}")
    return vs

def main():
    try:
        cfg = read_yaml(Path(CONFIG_PATH))
        logging.info(f"Loaded config from {CONFIG_PATH}")

        try:
            logging.setLevel(cfg.log_level)
        except Exception:
            pass

        load_env_from_config(cfg)

        txt_folder = os.path.normpath(cfg.paths.txt_folder)
        vectorstore_path = os.path.normpath(cfg.paths.vectorstore_path)

        glob_pattern = cfg.text.glob_pattern
        chunk_size = int(cfg.text.chunk_size)
        chunk_overlap = int(cfg.text.chunk_overlap)

        embedding_model = cfg.model.embedding_model
        collection_name = cfg.vectorstore.collection_name

        db_file = os.path.join(vectorstore_path, "chroma.sqlite3")
        if os.path.exists(db_file):
            logging.info(f"Vectorstore already exists at '{vectorstore_path}'. Skipping creation.")
            print("✅ Vectorstore already exists. Skipping creation.")
            return

        documents = load_documents(txt_folder, glob_pattern)
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        embeddings = build_embeddings(embedding_model)
        
        vectorstore = create_and_persist_vectorstore(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=collection_name,
            persist_directory=vectorstore_path
        )

        if os.path.exists(db_file):
            logging.info("Vectorization completed successfully.")
            print(f"✅ Vectorstore created with {len(chunks)} chunks and saved to: {vectorstore_path}")
        else:
            raise RuntimeError("Vectorstore file was not created!")

    except Exception as e:
        logging.error("Pipeline failed.", exc_info=True)
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
