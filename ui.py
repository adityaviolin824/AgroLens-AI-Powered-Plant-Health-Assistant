from pathlib import Path
import gradio as gr

from final_pipeline import gradio_pipeline, show_chat_and_init, chat

TEMPLATES_DIR = Path("templates")


def _read_template(name: str) -> str:
    p = TEMPLATES_DIR / name
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def create_demo():
    css = """
    /* Font import */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant:wght@600;700&display=swap');
    
    /* Colors */
    :root {
      --deep-blue: #071b2e;
      --deep-blue-2: #0b3350;
      --golden: #d9b074;
      --golden-glow: rgba(215,175,110,0.25);
      --text: #f3e9d3;
    }
    
    /* Base styles */
    body {
      background: linear-gradient(160deg, var(--deep-blue), var(--deep-blue-2) 45%, #b77a2b 110%);
      background-attachment: fixed;
      font-family: "Cormorant", serif;
      font-weight: 600;
      color: var(--text);
    }
    
    /* Gradio container with background image */
    .gradio-container {
      font-family: "Cormorant", serif;
      color: var(--text);
      background-image: url('/gradio_api/file=templates/agrolens_background.png') !important;
      background-size: cover !important;
      background-position: center !important;
      background-repeat: no-repeat !important;
    }
    
    /* Headings */
    h1, h2, h3 {
      font-family: "Cormorant", serif;
      font-weight: 700;
      color: var(--golden);
      text-shadow: 0 0 14px var(--golden-glow);
    }
    
    /* Panels */
    .gr-block, .gr-accordion, .gr-image, .gr-chatbot {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 16px;
      backdrop-filter: blur(6px);
    }
    """
    
    intro_html = _read_template("intro.html")
    footer_html = _read_template("footer.html")

    with gr.Blocks(title="AgroLens — AI-Powered Plant Health Assistant", css=css, theme=gr.themes.Soft()) as demo:
        if intro_html:
            gr.HTML(intro_html)
        else:
            print("NEED FOOTER")

        with gr.Accordion("ℹ️ How it works", open=False):
            gr.Markdown(
                """
                1. Upload Image
                2. AI Analysis
                3. Expert Advice
                4. Ask Questions
                """
            )

        # STAGE 1 & 2 UI
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    label="📸 Upload Plant Image",
                    type="filepath",
                    sources=["upload", "clipboard", "webcam"],
                    height=350
                )

                run_btn = gr.Button("🚀 Analyze Image", variant="primary", size="lg", elem_classes="primary-btn")

            with gr.Column(scale=1):
                output_md = gr.Markdown(
                    "<span style='color: gold;'>👆 Upload an image above and click <b>Analyze Image</b> to get started.</span>",
                    label="Results"
                )

        assist_btn = gr.Button(
            "💬 Need Further Assistance? Ask Questions",
            visible=False,
            variant="secondary",
            size="lg"
        )

        # STAGE 3 Chat (hidden by default)
        with gr.Group(visible=False) as chat_section:
            gr.Markdown("### 💬 Ask Follow-up Questions")
            gr.Markdown("Get personalized answers about your plant's condition, treatment options, prevention tips, and more.")

            chatbot = gr.Chatbot(
                label="🤖 AgroLens Assistant",
                height=450,
                show_label=True,
                avatar_images=(None, "templates/bot_avatar.png"),
                bubble_full_width=False
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="",
                    placeholder="💭 Ask about treatment methods, prevention strategies, organic solutions...",
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("📤 Send", variant="primary", scale=1)

            with gr.Accordion("💡 Example Questions", open=False):
                gr.Markdown(
                    """
                    - How can I treat this organically?
                    - Can this spread to other plants?
                    - What preventive measures should I take?
                    """
                )

            clear_chat_btn = gr.Button("🗑️ Clear Conversation", variant="secondary", size="sm")

        conversation_chain = gr.State(None)

        # Footer
        if footer_html:
            gr.HTML(footer_html)
        else:
            print("NEED FOOTER")

        # EVENT HANDLERS
        run_btn.click(
            fn=gradio_pipeline,
            inputs=[image_in],
            outputs=[output_md, assist_btn],
            show_progress="full",
        )

        assist_btn.click(
            fn=show_chat_and_init,
            inputs=None,
            outputs=[chat_section, conversation_chain]
        )

        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot, conversation_chain],
            outputs=chatbot
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=msg_input
        )

        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot, conversation_chain],
            outputs=chatbot
        ).then(
            fn=lambda: "",
            inputs=None,
            outputs=msg_input
        )

        clear_chat_btn.click(
            fn=lambda: [],
            inputs=None,
            outputs=[chatbot]
        )

    demo.queue()
    
    return demo
