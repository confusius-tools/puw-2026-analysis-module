# PUW 2026 — Analysis module

Marimo slides for the **Paris Ultrasound Workshop 2026** module
**Analysis & open-source tools**.

## Start editing

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked
uv run marimo edit slides.py
```

Open the app preview to see the slides layout. Each cell with visible output
becomes a slide by default. Use the layout sidebar to set slide types and
speaker notes; marimo saves these settings in `layouts/`, which must stay in Git.

## Present

```sh
uv run marimo run slides.py
```

Navigate with the arrow keys; press `F` for fullscreen and `S` for speaker view.

## Export a static deck

```sh
uv run marimo export html slides.py -o build/slides.html
```

Static HTML preserves rendered outputs, not live Python interactivity.
Speaker notes are embedded in the export: do not put private information in them.

## Check

```sh
uv run marimo check slides.py
```

`slides.py` contains a draft outline, not finished teaching material. Add analysis
packages with `uv add` when the worked example needs them. No dataset is bundled;
only add data you have permission to share.
