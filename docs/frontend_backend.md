# Retina Frontend + Backend

Retina is local-first. The backend exposes the frozen CLIP backbone, trainable query adapter, and FAISS search service, and the React dashboard consumes only JSON responses plus browser-safe image URLs.

## Local commands

Backend:

```bash
make api
```

Frontend setup:

```bash
make frontend-install
```

Frontend dev server:

```bash
make frontend
```

If you prefer to run Vite directly:

```bash
cd frontend
npm install
npm run dev
```

## API contract

- `GET /health`
- `GET /metadata`
- `GET /metrics/summary`
- `GET /recommend/text?query=...&top_k=...`
- `GET /recommend/image?image_id=...&top_k=...`
- `POST /recommend/profile`
- `POST /search/text` for backward compatibility
- `GET /artifacts/images/{relative_path}` for safe local artifact serving

## Response fields

Recommendation responses include:

- `service`
- `top_k`
- `latency_ms`
- `results`

Each result includes:

- `image_id`
- `image_path`
- `image_url` when the backend can safely serve the local artifact
- `captions`
- `caption`
- `score`
- `rank`
- `recommendation_reason`

## CORS

The backend allows the React dev server origin by default:

- `http://localhost:5173`

You can override the origin list in `configs/retina.yaml`.

## Static images

The backend only serves files under the configured artifact image directory.
Raw images, embeddings, FAISS indexes, and model weights stay local and are not committed.

If the browser cannot render a local `image_path`, use the `image_url` returned by the API or configure the backend to serve the artifact directory safely.
