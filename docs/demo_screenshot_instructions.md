# Retina Demo Screenshot Instructions

Do not fabricate a screenshot. Capture the real local dashboard after the backend and frontend are running.

## Exact steps

1. Start the backend:

```bash
make api
```

2. Start the frontend in a second terminal:

```bash
make frontend
```

3. Open the Vite URL printed in the terminal, usually `http://localhost:5173`.
4. Capture the Overview tab first so the metrics cards are visible.
5. Try a representative text query such as:

```text
A bicyclist doing a jump trick
```

6. Capture the Text Recommendations tab with real results.
7. Capture the Research Results tab with the benchmark tables visible.
8. Optionally capture the Similar Images and Profile Recommendations tabs if you want a fuller product walkthrough.
9. Save the screenshots at:

```text
docs/assets/retina_dashboard_overview.png
docs/assets/retina_text_recommendations.png
docs/assets/retina_research_results.png
```

10. Link them from the README with:

```markdown
![Retina overview](docs/assets/retina_dashboard_overview.png)
![Retina text recommendations](docs/assets/retina_text_recommendations.png)
![Retina research results](docs/assets/retina_research_results.png)
```

## Notes

- If the browser cannot render a local `image_path`, rely on the API-provided `image_url` field.
- Keep raw images, embeddings, and FAISS indexes uncommitted.
- The screenshot should show the dashboard, backend metadata, and at least one populated results grid.
