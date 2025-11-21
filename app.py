from ui import create_demo

if __name__ == "__main__":
    demo = create_demo()
    demo.queue()
    demo.launch(allowed_paths=["templates/agrolens_background.png"])
