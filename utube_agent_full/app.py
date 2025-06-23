import gradio as gr
from backend_with_chain import get_answer_from_video

def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    else:
        return ""

def run_agent(youtube_url, question):
    try:
        answer = get_answer_from_video(youtube_url, question)
        return answer
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(css="""
body {
    background: linear-gradient(to bottom right, #0f2027, #203a43, #2c5364);
    font-family: 'Segoe UI', sans-serif;
}
.gradio-container {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 );
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: white;
    margin-top: 30px;
}
.gr-button {
    background-color: #1e88e5 !important;
    color: white !important;
    border-radius: 12px !important;
}
.gr-button:hover {
    background-color: #1565c0 !important;
}
input, textarea {
    background-color: #0f1117 !important;
    color: white !important;
}
""") as demo:

    gr.Markdown("## 🎬 Ask Your YouTube Agent", elem_classes="title")
    gr.Markdown(
        "Paste a YouTube link and ask your question. The agent fetches transcript, embeds info, retrieves context, and answers using **Mistral (Ollama)** locally.",
        elem_classes="subtitle"
    )

    with gr.Row():
        youtube_url = gr.Textbox(label="📺 YouTube URL", placeholder="https://youtube.com/watch?v=...")
        question = gr.Textbox(label="💬 Your Question", placeholder="What is it about?")

    thumbnail = gr.Image(label="🎞️ Video Thumbnail", height=180, width=320, visible=False)

    with gr.Row():
        run_btn = gr.Button("🚀 Run Agent")

    answer_box = gr.Textbox(label="🧠 Agent's Answer", lines=6, interactive=False)

    def on_submit(url, q):
        video_id = extract_video_id(url)
        if video_id:
            thumb_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
            answer = run_agent(url, q)
            return answer, gr.update(value=thumb_url, visible=True)
        else:
            return "❌ Invalid YouTube URL", gr.update(visible=False)

    run_btn.click(
        on_submit,
        inputs=[youtube_url, question],
        outputs=[answer_box, thumbnail]
    )

demo.launch()
