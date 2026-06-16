# Retina Demo Screenshot Instructions

If you want to add a real demo screenshot later, use the local Gradio app and capture the rendered UI yourself.

## Exact steps

1. Run the demo:

```bash
make demo
```

2. Open the local Gradio URL printed in the terminal.
3. Use a representative text query such as:

```text
A bicyclist doing a jump trick
```

4. Capture the page after results render.
5. Save the image at:

```text
docs/assets/retina_demo.png
```

6. If you add the screenshot, update the README image link to point at:

```markdown
![Retina demo](docs/assets/retina_demo.png)
```

## Notes

- Do not use a fabricated or synthetic screenshot.
- The UI should show text-to-image, image-to-image, and profile recommendation tabs before taking the capture.

