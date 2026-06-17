# Retina UI Demo

## What the UI shows

- A browser-first landing page with text, image, and profile search modes.
- A result inspector that exposes score, rank, captions, and metadata for the selected card.
- A local search history panel so demo sessions can be replayed without manual notes.

## Demo flow

1. Open the app with `make frontend` and `make api`.
2. Run the sample text query from the hero section.
3. Click any result card to inspect why it ranked where it did.
4. Switch to image or profile mode to compare result quality across modes.

## Notes

- The UI uses only local query and retrieval outputs.
- Result explanations are derived from returned score, rank, captions, and metadata fields.
