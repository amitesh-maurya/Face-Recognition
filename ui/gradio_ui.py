import gradio as gr
from core.config import TITLE, DESC
from core.face_utils import recognize_image, enroll_image

def build_ui():
    with gr.Blocks(title=TITLE) as demo:
        gr.Markdown(f"# {TITLE}\n\n{DESC}")

        # Recognize Tab
        with gr.Tab("Recognize"):
            inp = gr.Image(label="Image / Webcam", sources=["upload", "webcam"], type="numpy")
            out = gr.Image(label="Output")
            # Process on image change
            inp.change(fn=recognize_image, inputs=inp, outputs=out)

        # Enroll Tab
        with gr.Tab("Enroll"):
            enroll_img = gr.Image(label="Image with face", sources=["upload", "webcam"], type="numpy")
            person = gr.Textbox(label="Person ID")
            enroll_btn = gr.Button("Enroll")
            status = gr.Textbox(label="Status")
            enroll_out = gr.Image(label="Preview")
            enroll_btn.click(fn=enroll_image, inputs=[enroll_img, person], outputs=[status, enroll_out])

    return demo
