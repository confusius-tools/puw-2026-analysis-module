import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="PUW 2026 — fUSI analysis & open-source tools",
    layout_file="layouts/slides.slides.json",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Analysis & open-source tools

    ## Paris Ultrasound Workshop · 2026-12-09
    """)
    return


if __name__ == "__main__":
    app.run()
