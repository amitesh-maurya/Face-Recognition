from ui.gradio_ui import build_ui
from core.gallery import load_gallery
from core import load_gallery
from ui import build_ui

if __name__ == "__main__":
    load_gallery()
    demo = build_ui()
    demo.launch()
