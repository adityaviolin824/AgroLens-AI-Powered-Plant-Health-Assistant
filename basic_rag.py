import io
import locale
import sys
import os

import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler

from utils.read_yaml import read_yaml
from utils.logger import logging
from utils.exception import CustomException

import warnings
warnings.filterwarnings("ignore")


CONFIG_PATH = Path("configuration-files/rag_config.yaml")


def load_env_from_config(cfg):
    dotenv_override = bool(cfg.env.dotenv_override)
    load_dotenv(override=dotenv_override)
    key_name = cfg.env.openai_key_env
    os.environ["OPENAI_API_KEY"] = os.getenv(key_name, "your-key-if-not-using-env")
    logging.info("Environment variables loaded.")

def build_rag_chain(cfg):
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    chat_model      = cfg.model.chat_model
    embedding_model = cfg.model.embedding_model
    temperature     = float(cfg.llm.temperature)
    search_k        = int(cfg.vectorstore.search_k)
    collection_name = cfg.vectorstore.collection_name
    persist_dir     = Path(cfg.paths.vectorstore_path)

    if not persist_dir.exists():
        msg = (
            f"Vectorstore directory not found: {persist_dir}. "
            "Run your vectorstore creation script first."
        )
        logging.error(msg)
        raise CustomException(msg, sys)

    embeddings = OpenAIEmbeddings(model=embedding_model)
    logging.info(f"Loaded embedding model: {embedding_model}")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
    logging.info(f"Loaded existing vectorstore from: {persist_dir}")

    llm = ChatOpenAI(
        temperature=temperature,
        model_name=chat_model,        
        max_tokens=cfg.llm.get("max_tokens", 800),
        streaming=cfg.llm.get("streaming", False),
    )
    logging.info(f"Chat model initialized: {chat_model}")

    memory = ConversationBufferMemory(
        memory_key=cfg.memory.memory_key,
        return_messages=cfg.memory.return_messages,
    )
    logging.info("Conversation memory created.")

    custom_prompt = PromptTemplate(
        input_variables=cfg.prompt.input_variables,
        template=cfg.prompt.template,
    )
    logging.info("Custom prompt template loaded.")

    callbacks = [StdOutCallbackHandler()] if cfg.llm.callbacks_enabled else []
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        callbacks=callbacks,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
    )

    logging.info("Conversational retrieval chain created successfully.")
    return conversation_chain


def main():
    try:
        cfg = read_yaml(CONFIG_PATH)
        logging.info(f"Loaded config from {CONFIG_PATH}")

        try:
            logging.setLevel(cfg.log_level)
        except Exception:
            pass

        load_env_from_config(cfg)

        conversation_chain = build_rag_chain(cfg)

        predicted_label = None
        try:
            if cfg.output.save_enabled:
                pred_json_path = Path(cfg.output.save_dir) / cfg.output.file_name
                if pred_json_path.exists():
                    with open(pred_json_path, "r", encoding="utf-8") as f:
                        predicted_label = json.load(f).get("predicted_label")
                    logging.info(f"Loaded predicted label from {pred_json_path}: {predicted_label}")
        except Exception:
            logging.warning("Failed to read predicted label JSON; proceeding without it.", exc_info=True)

        if predicted_label:
            logging.info(f"Passing in the predicted condition << {predicted_label} >> into our LLM for insights")

            if predicted_label.lower() == "healthy":
                query = (
                    "The image suggests the plant is healthy. "
                    "Open with a brief, positive note for the farmer. "
                    "Name 3–4 visible signs of good health. "
                    "Give 3–4 simple tips to keep it that way (watering, sunlight, soil, pest prevention). "
                    "Be friendly and practical."
                )
            else:
                query = (
                    f"The disease detected is {predicted_label}. "
                    "Start by stating this clearly. "
                    "In one short line, explain what it means for the crop. "
                    "List 4–6 concrete steps: remove or isolate affected parts, how to treat, how to prevent spread, and what to monitor next. "
                    "Offer a low-cost option if possible. "
                    "Keep the tone calm, supportive, and farmer-friendly."
                )
        else:
            query = (
                "A plant problem is likely but the disease was not identified. "
                "Give a short checklist to recognise common diseases (spots, mildew, rust, blight) and 4–6 preventive steps. "
                "Suggest when to seek local expert help. "
                "Keep it friendly and practical."
            )

        result = conversation_chain.invoke({"question": query})
        answer_text = result["answer"]

        response_path = Path(cfg.conversation.last_response_path)
        response_path.parent.mkdir(parents=True, exist_ok=True)
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(answer_text)
        logging.info(f"Latest RAG response saved to: {response_path}")

        conversation_history = "\n".join(
            [f"{m.type.upper()}: {m.content}" for m in conversation_chain.memory.chat_memory.messages]
        )

        print("\nAnswer:", answer_text)
        logging.info("RAG chat completed successfully.")

        if cfg.conversation.save_enabled:
            try:
                save_path = Path(cfg.conversation.memory_file_name)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write("=== Conversation History ===\n")
                    f.write(conversation_history + "\n\n")
                logging.info(f"Conversation saved to: {save_path}")
            except Exception as e:
                logging.error(f"Failed to save conversation: {e}")

    except Exception as e:
        logging.error("RAG Chatbot pipeline failed.", exc_info=True)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
