# ----------------------------
# Gradio front-end for Stage 1 + Stage 2 + Stage 3 (Chat)
# ----------------------------


import gradio as gr
from pathlib import Path
from PIL import Image
import time
import io
import os
import locale
import sys

from utils.logger import logging
from cv_inference import load_cfg, run_inference
from basic_rag import main as run_rag_main
from start_chatbot import build_conversational_chain


# ----------------------------
# Core Pipeline Function
# ----------------------------
def run_pipeline(image_path: Path):
    """
    Run Stage 1 (CV prediction) and Stage 2 (RAG).
    Args:
        image_path: Path to the input image
    Returns:
        str: The predicted label from Stage 1
    """
    logging.info("<<< Starting AgroLens Pipeline >>>")
    start_time = time.time()

    # Load YAML config for CV inference
    logging.info("<<< Loading YAML configuration for CV inference >>>")
    cfg = load_cfg()

    # ---------------------------
    # Stage 1
    # ---------------------------
    stage1_start = time.time()
    logging.info("<<< Stage 1: Starting inference on image >>>")

    pred_label = run_inference(image_path=image_path, cfg=cfg)

    stage1_time = time.time() - stage1_start
    logging.info(f"<<< Stage 1 Complete: Predicted label = '{pred_label}' | Time taken: {stage1_time:.2f}s >>>")

    # ---------------------------
    # Stage 2
    # ---------------------------
    stage2_start = time.time()
    logging.info("<<< Stage 2: Running RAG Chatbot >>>")

    run_rag_main()

    stage2_time = time.time() - stage2_start
    logging.info(f"<<< Stage 2 Complete | Time taken: {stage2_time:.2f}s >>>")

    total_time = time.time() - start_time
    logging.info(f"<<< Pipeline Complete | Total runtime: {total_time:.2f}s >>>")

    return pred_label


def gradio_pipeline(image_filepath: str):
    """
    Generator that yields intermediate and final outputs for Gradio.
    Each yield is a tuple: (markdown_text, assist_button_update)
    """
    if not image_filepath or not Path(image_filepath).exists():
        yield "❌ No image provided or file not found.", gr.update(visible=False)
        return

    yield (
        "⏳ Running CV inference and analysis. Please wait...",
        gr.update(visible=False),
    )

    # Run stage 1 & 2
    pred_label = run_pipeline(image_path=Path(image_filepath))

    llm_output = ""
    response_path = Path("pipeline-files/last_rag_response.txt")
    if response_path.exists():
        with response_path.open("r", encoding="utf-8") as f:
            llm_output = f.read().strip()

    # Final output (NO STREAMING - just show complete result)
    if llm_output:
        formatted_llm = llm_output.replace('\n', '<br>')
        
        final_text = f"""<div style='color: #f3e9d3; font-family: Cormorant, serif;'>
    <h3 style='color: #d9b074; text-shadow: 0 0 14px rgba(215,175,110,0.25);'>🌿 Predicted Condition: <span style='background: rgba(217,176,116,0.2); padding: 2px 8px; border-radius: 4px;'>{pred_label}</span></h3>
    <hr style='border: 1px solid rgba(217,176,116,0.3); margin: 16px 0;'>
    <h4 style='color: #d9b074;'>📋 Analysis and Tips for Further Action:</h4>
    <div style='color: #f3e9d3; line-height: 1.6; font-weight: bold;'>{formatted_llm}</div>
    </div>"""
        
        yield (final_text, gr.update(visible=False))
        time.sleep(2.5)
        yield (final_text, gr.update(visible=True))
    else:
        final_text = f"""<div style='color: #f3e9d3; font-family: Cormorant, serif;'>
    <h3 style='color: #d9b074; text-shadow: 0 0 14px rgba(215,175,110,0.25);'>🌿 Predicted Condition: <span style='background: rgba(217,176,116,0.2); padding: 2px 8px; border-radius: 4px;'>{pred_label}</span></h3>
    <hr style='border: 1px solid rgba(217,176,116,0.3); margin: 16px 0;'>
    <p style='color: #ff9999; font-weight: bold;'>⚠️ RAG Response not found or empty.</p>
    </div>"""
        
        yield (final_text, gr.update(visible=False))




# ----------------------------
# Stage 3: Chat Functions
# ----------------------------
def show_chat_and_init():
    """
    Initialize chat when 'Need Further Assistance?' button is clicked.
    
    Builds the conversational chain which automatically:
    - Loads embeddings and vectorstore
    - Reads conversation history from file (saved by Stage 2)
    - Sets up memory with previous context
    
    Returns:
        tuple: (chat_group_visibility, chain_instance)
    """
    chain = build_conversational_chain("configuration-files/follow_up_chat_config.yaml")
    
    logging.info("Stage 3 chat initialized")
    return gr.Group(visible=True), chain


# ----------------------------
# Stage 3 Chat with STREAMING
# ----------------------------
def chat(user_message: str, history: list, chain):
    """
    Handle chat messages in Stage 3 with streaming.
    
    Args:
        user_message: User's question
        history: Chat history as list of tuples [(user, bot), ...]
        chain: The ConversationalRetrievalChain from State
        
    Yields:
        Updated history with streaming response
    """
    if not user_message.strip():
        yield history
        return
    
    try:
        result = chain.invoke({"question": user_message})
        answer = result.get("answer", "")
        
        logging.info(f"Follow-up Q: {user_message[:50]}...")
        
    except Exception as e:
        logging.error(f"Chat error: {e}", exc_info=True)
        answer = f"❌ Error processing your question: {str(e)}"
    
    history = history + [(user_message, "")]
    
    for i in range(len(answer)):
        history[-1] = (user_message, answer[:i+1])
        yield history
        time.sleep(0.01)  
    
    yield history