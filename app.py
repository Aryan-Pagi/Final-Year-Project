import gradio as gr

from gradio_backend import (
    start_backend,
    stop_backend,
    clear_prediction,
    backend_status,
)

CSS = """
body{
    background:#0b0b0b;
}

.gradio-container{
    background:#0b0b0b !important;
}

.header{
    background:#111;
    border:1px solid #2b2b2b;
    padding:12px;
    border-radius:10px;
    margin-bottom:10px;
}

.output-box{
    border:1px solid #333;
    border-radius:10px;
    padding:15px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

.side-btn{
    width:100%;
}
"""


def launch():

    with gr.Blocks(css=CSS, title="ISL Translator") as demo:

        with gr.Row(elem_classes="header"):

            gr.Markdown(
                "## ISL TRANSLATOR | ACTIVE ONLINE"
            )

            status = gr.Textbox(
                value=backend_status(),
                interactive=False,
                label="Status"
            )

        with gr.Row():

            # ---------------- LEFT ---------------- #

            with gr.Column(scale=4):

                camera = gr.Image(
                    label="Live Camera",
                    interactive=False,
                    height=520
                )

                prediction = gr.Textbox(
                    value="Waiting...",
                    label="Prediction",
                    elem_classes="output-box",
                    interactive=False
                )

                confidence = gr.Textbox(
                    value="0%",
                    label="Confidence",
                    interactive=False
                )

            # ---------------- RIGHT ---------------- #

            with gr.Column(scale=1):

                admin = gr.Button("ADMIN")

                logs = gr.Button("LOGS")

                settings = gr.Button("SET")

                gr.Markdown("---")

                start = gr.Button(
                    "START",
                    variant="primary"
                )

                stop = gr.Button(
                    "STOP",
                    variant="stop"
                )

                recognizing = gr.Button(
                    "🟢 RECOGNIZING",
                    interactive=False
                )

                clear = gr.Button(
                    "CLEAR"
                )

        # ---------------- EVENTS ---------------- #

        start.click(
            fn=start_backend,
            outputs=status
        )

        stop.click(
            fn=stop_backend,
            outputs=status
        )

        clear.click(
            fn=clear_prediction,
            outputs=[
                prediction,
                confidence
            ]
        )

    demo.launch(
        server_name="127.0.0.1",
        server_port=5000,
        inbrowser=True,
        share=False
    )


if __name__ == "__main__":
    launch()