from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import gradio as gr

from retrieval.search import RetinaSearchEngine


def build_demo(engine: RetinaSearchEngine) -> gr.Blocks:
    with gr.Blocks(title="Retina") as demo:
        gr.Markdown("# Retina\nVisual intelligence search engine")
        query = gr.Textbox(label="Text query")
        top_k = gr.Slider(1, 20, value=5, step=1, label="Top-k")
        gallery = gr.Gallery(label="Results", columns=2, height=320)
        table = gr.Dataframe(label="Results metadata")

        def search(query_text: str, k: int):
            results = engine.search_text(query_text, top_k=int(k))
            images = []
            rows = []
            for item in results:
                images.append((item["image_path"], f'{item.get("caption", "")} | score={item["score"]:.3f}'))
                rows.append([item.get("rank"), item.get("image_id"), item.get("score"), item.get("caption")])
            return images, rows

        button = gr.Button("Search")
        button.click(search, inputs=[query, top_k], outputs=[gallery, table])
    return demo

