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
4. Try a representative text query such as:

```text
A bicyclist doing a jump trick
```

5. Also check the profile and similar-image tabs so the screenshot shows the full product shape.
6. Save the screenshot at:

```text
docs/assets/retina_dashboard.png
```

7. Link it from the README with:

```markdown
![Retina dashboard](docs/assets/retina_dashboard.png)
```

## Notes

- If the browser cannot render a local `image_path`, rely on the API-provided `image_url` field.
- Keep raw images, embeddings, and FAISS indexes uncommitted.
- The screenshot should show the dashboard, backend metadata, and at least one populated results grid.
