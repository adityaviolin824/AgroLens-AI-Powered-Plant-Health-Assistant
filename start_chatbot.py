"""
Minimal ChatChainService with fail-fast config loading.
All config values are required - crashes immediately if missing.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv
load_dotenv(override=True)

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


def load_and_inject_memory(memory, path: Path) -> int:
    """Load HUMAN/AI conversation file and inject into memory."""
    if not path.exists():
        logging.info("No conversation file at %s", path)
        return 0
    
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return 0

    patt = re.compile(
        r"^(HUMAN|AI)\s*:\s*(.+?)(?=(?:\n(?:HUMAN|AI)\s*:)|\Z)",
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    
    pairs: List[Tuple[str, str]] = []
    for match in patt.finditer(raw):
        role = "user" if match.group(1).lower() == "human" else "assistant"
        text = match.group(2).strip()
        if text:
            pairs.append((role, text))

    # Inject into memory
    for role, text in pairs:
        if role == "user":
            memory.chat_memory.add_user_message(text)
        else:
            memory.chat_memory.add_ai_message(text)

    logging.info("Loaded %d messages", len(pairs))
    return len(pairs)


class ChatChainService:
    """Minimal service that builds ConversationalRetrievalChain from config.
    Fails immediately if any required config is missing.
    """

    def __init__(self, cfg_path: str | Path):
        """Initialize with config path. Fails if config invalid or missing required values."""
        self.cfg_path = Path(cfg_path)
        
        # Load config
        cfg = read_yaml(self.cfg_path)
        if not isinstance(cfg, dict):
            raise ValueError(f"Config at {self.cfg_path} is not a dict")
        self.cfg: Dict[str, Any] = cfg

        # Extract all paths
        paths = self.cfg["paths"]
        self.conv_path = Path(paths["conversation_memory"])
        self.vectorstore_dir = Path(paths["vectorstore_dir"])
        if not self.vectorstore_dir.exists():
            raise CustomException(f"Vectorstore dir not found: {self.vectorstore_dir}", sys)

        # Extract LLM config - fail if missing
        llm_cfg = self.cfg["llm"]
        self.llm_model = llm_cfg["model_name"]
        self.llm_temperature = llm_cfg["temperature"]
        self.llm_max_tokens = llm_cfg["max_tokens"]

        # Extract embeddings config
        self.embeddings_model = self.cfg["embeddings"]["model_name"]

        # Extract retriever config
        self.retriever_k = int(self.cfg["retriever"]["k"])

        # Extract prompt template
        self.prompt_template = self.cfg["prompt"]["template"]

        mem_cfg = self.cfg["memory"]
        self.memory_key = mem_cfg["memory_key"]
        self.return_messages = bool(mem_cfg["return_messages"])

        self.enable_stdout = bool(self.cfg["callbacks"]["enable_stdout"])

        logging.info("Initializing embeddings: %s", self.embeddings_model)
        self.embeddings = OpenAIEmbeddings(model=self.embeddings_model)

        logging.info("Loading vectorstore: %s", self.vectorstore_dir)
        self.vectorstore = Chroma(
            collection_name="plant_docs",
            embedding_function=self.embeddings,
            persist_directory=str(self.vectorstore_dir),
        )

        logging.info("ChatChainService initialized")

    def build_chain(self) -> ConversationalRetrievalChain:
        """Build and return ConversationalRetrievalChain."""
        # LLM
        llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )

        memory = ConversationBufferMemory(
            memory_key=self.memory_key,
            return_messages=self.return_messages,
            output_key='answer'
        )
        load_and_inject_memory(memory, self.conv_path)

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.retriever_k})

        qa_prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=self.prompt_template,
        )
        
        passthrough_prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template="{question}"
        )

        callbacks = [StdOutCallbackHandler()] if self.enable_stdout else []

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            callbacks=callbacks,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            condense_question_prompt=passthrough_prompt,
            rephrase_question=False,
        )
        
        logging.info("Chain built")
        return chain


def build_conversational_chain(cfg_path: str | Path) -> ConversationalRetrievalChain: 
    """Factory function to build chain from config path."""
    svc = ChatChainService(cfg_path)
    return svc.build_chain()


if __name__ == "__main__":
    import gradio as gr
    import time

    chain = build_conversational_chain("configuration-files/follow_up_chat_config.yaml")

    def chat_fn(user_message, history):
        """Chat function with fake streaming (gets full response, then streams to UI)."""
        if not user_message.strip():
            yield history 
            return

        try:
            result = chain.invoke({"question": user_message})
            answer = result.get("answer", "")
        except Exception as e:
            answer = f"[Error: {str(e)}]"

        history = history + [(user_message, "")]
        
        for i in range(len(answer)):
            history[-1] = (user_message, answer[:i+1])
            yield history
            time.sleep(0.01) 
        
        yield history

    with gr.Blocks(title="Follow-up Chat Test") as demo:
        gr.Markdown("## Quick Test Chat for Follow-up LLM (Streaming)")

        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(placeholder="Type a message...")
        send_btn = gr.Button("Send")

        send_btn.click(chat_fn, inputs=[msg, chatbot], outputs=chatbot)
        msg.submit(chat_fn, inputs=[msg, chatbot], outputs=chatbot)

    demo.launch()
