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


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
            # Today’s roadmap

            From understanding the signal to sharing the science.
            """),
            mo.hstack(
                [
                    mo.callout(mo.md("""
                    ## 01 · Foundations

                    **Understand the signal**

                    Concepts, terminology & analysis choices.
                    """)),
                    mo.callout(mo.md("""
                    ## 02 · Hands-on pipeline

                    **Turn data into results**

                    Acquisition, preparation & statistical analysis.
                    """)),
                    mo.callout(mo.md("""
                    ## 03 · Go open

                    **Build on shared tools**

                    Open-source software, datasets & community.
                    """)),
                ],
                widths="equal",
                align="stretch",
                gap=1.5,
            ),
        ],
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 1 · Foundations

    ## What do we measure, and how do we analyse it?

    - **Concepts:** neurovascular coupling, task-based fUSI, resting-state connectivity & multimodal imaging
    - **Before analysis:** acquisition, hardware, experimental design & neuronavigation
    - **Analysis choices:** fUSI-BIDS standardisation → registration → denoising → statistics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 2 · Hands-on pipeline

    ## From real data to interpretable results

    - **Acquire & organise:** ICONEUS One / Verasonics + EffUSIve → fUSI-BIDS
    - **Prepare:** registration & denoising with ConfUSIus
    - **Analyse:** task-based GLM (human), multimodal imaging (rat & NHP), resting-state connectivity (mouse)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 3 · Go open

    ## Reuse tools. Share data. Contribute back.

    - **Connect the tools:** Echoframe, EffUSIve, fUSI-BIDS, ConfUSIus, OfUSA, PyfUS, Pepe fMRI, fUSI-RABIES & NIPY
    - **Share datasets:** make analyses reproducible and comparisons possible
    - **Join in:** report issues, improve documentation & contribute workflows
    """)
    return


if __name__ == "__main__":
    app.run()
