from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import gradio as gr

from retrieval.search import RetinaSearchEngine


def build_demo(engine: RetinaSearchEngine) -> gr.Blocks:
    with gr.Blocks(title="Retina") as demo:
        gr.Markdown(
            "# Retina\nContent-based visual recommendation system using CLIP embedding similarity.\n\n"
            "Not trained on user behavior. Not collaborative filtering."
        )

        def format_results(results):
            images = []
            rows = []
            for item in results:
                label = f'{item.get("caption", "")} | score={item["score"]:.3f}'
                images.append((item["image_path"], label))
                rows.append(
                    [
                        item.get("rank"),
                        item.get("image_id"),
                        item.get("score"),
                        item.get("recommendation_reason", ""),
                        item.get("caption", ""),
                    ]
                )
            return images, rows

        recommend_text = getattr(engine, "recommend_text", None)
        if recommend_text is None:
            recommend_text = engine.search_text
        recommend_image = getattr(engine, "recommend_image", None)
        if recommend_image is None:
            recommend_image = getattr(engine, "search_image", None)
        if recommend_image is None:
            recommend_image = lambda *args, **kwargs: []
        profile_recommend = getattr(engine, "recommend_profile", None)

        with gr.Tab("Text Search"):
            text_query = gr.Textbox(label="Query text")
            text_top_k = gr.Slider(1, 20, value=5, step=1, label="Top-k")
            text_gallery = gr.Gallery(label="Results", columns=2, height=320)
            text_table = gr.Dataframe(label="Results metadata")

            def search_text(query_text: str, k: int):
                results = recommend_text(query_text, top_k=int(k))
                return format_results(results)

            gr.Button("Recommend").click(search_text, inputs=[text_query, text_top_k], outputs=[text_gallery, text_table])

        with gr.Tab("Similar Images"):
            image_id = gr.Textbox(label="Image ID")
            image_file = gr.Image(label="Or upload an image", type="filepath")
            image_top_k = gr.Slider(1, 20, value=5, step=1, label="Top-k")
            image_gallery = gr.Gallery(label="Results", columns=2, height=320)
            image_table = gr.Dataframe(label="Results metadata")

            def search_image(image_id_value: str, uploaded_image: str | None, k: int):
                if image_id_value.strip():
                    results = recommend_image(image_id=image_id_value.strip(), top_k=int(k))
                elif uploaded_image:
                    results = recommend_image(image_path=uploaded_image, top_k=int(k))
                else:
                    results = []
                return format_results(results)

            gr.Button("Find Similar").click(
                search_image,
                inputs=[image_id, image_file, image_top_k],
                outputs=[image_gallery, image_table],
            )

        with gr.Tab("Profile Recommendations"):
            interests = gr.Textbox(lines=4, label="Text interests, one per line")
            liked_ids = gr.Textbox(lines=2, label="Liked image IDs, comma separated")
            profile_top_k = gr.Slider(1, 20, value=5, step=1, label="Top-k")
            profile_gallery = gr.Gallery(label="Results", columns=2, height=320)
            profile_table = gr.Dataframe(label="Results metadata")

            def recommend_profile_text(interests_text: str, liked_ids_text: str, k: int):
                text_queries = [line.strip() for line in interests_text.splitlines() if line.strip()]
                liked_image_ids = [item.strip() for item in liked_ids_text.split(",") if item.strip()]
                if profile_recommend is None:
                    results = recommend_text(" ".join(text_queries), top_k=int(k))
                else:
                    results = profile_recommend(text_queries=text_queries, liked_image_ids=liked_image_ids, top_k=int(k))
                return format_results(results)

            gr.Button("Recommend From Profile").click(
                recommend_profile_text,
                inputs=[interests, liked_ids, profile_top_k],
                outputs=[profile_gallery, profile_table],
            )
    return demo
