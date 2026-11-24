import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="PUW 2026 — fUSI analysis & open-source tools",
    layout_file="layouts/slides.slides.json",
)


@app.cell
def _():
    import io
    import tempfile
    from pathlib import Path

    import confusius as cf
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    return (
        FuncAnimation,
        Line2D,
        Path,
        PillowWriter,
        Rectangle,
        cf,
        io,
        mo,
        np,
        plt,
        tempfile,
    )


@app.cell(hide_code=True)
def _(io, plt):
    # The deck theme is dark (pyproject: tool.marimo.display.theme), so figures are
    # transparent with light text.
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 17,
            "figure.facecolor": "none",
            "savefig.facecolor": "none",
        }
    )

    PANEL = "#0d0f0e"
    ACCENT = "#ffd43b"
    OUTLINE = "#4dd0e1"
    # The red and cyan of the fixed/moving overlays, a little lighter for text.
    FIXED = "#ff4d4d"
    MOVING = "#00e5ff"


    def fig_png(fig, dpi=140):
        """Render a figure to PNG bytes and close it."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi)
        plt.close(fig)
        return buffer.getvalue()

    return ACCENT, FIXED, MOVING, OUTLINE, PANEL, fig_png


@app.cell(hide_code=True)
def _(ACCENT, np):
    def affine_latex(affine, previous=None):
        """Write an affine as a LaTeX matrix.

        The entries that differ from `previous` get the accent color.
        """
        # Round away float noise; adding 0.0 turns -0.0 into 0.0.
        affine = np.round(affine, 3) + 0.0
        previous = affine if previous is None else np.round(previous, 3) + 0.0

        def entry(value, before):
            text = np.format_float_positional(value, trim="-")
            return text if value == before else rf"\textcolor{{{ACCENT}}}{{{text}}}"

        rows = r" \\ ".join(
            " & ".join(entry(value, before) for value, before in zip(row, row_before))
            for row, row_before in zip(affine, previous)
        )
        return r"\begin{bmatrix} " + rows + r" \end{bmatrix}"

    return (affine_latex,)


@app.cell(hide_code=True)
def _(mo):
    def fragment(content, index, kind="", style=""):
        """Wrap content in a reveal.js fragment, shown at key press `index` (from 0).

        Fragments with the same index change together. `kind` is an optional reveal.js
        fragment class, for example "current-visible".
        """
        return mo.Html(
            f'<div class="fragment {kind}" data-fragment-index="{index}" style="{style}">'
            f"{content.text}</div>"
        )

    return (fragment,)


@app.cell(hide_code=True)
def _(
    ACCENT,
    Line2D,
    OUTLINE,
    PANEL,
    Path,
    Rectangle,
    cf,
    fig_png,
    mo,
    np,
    plt,
):
    # Invented probe pose for the demo: tune these two and the frame follows.
    ROTATION_DEG = -15  # about z, around the world origin
    SHIFT_MM = (0.5, 1.0, 2.5)  # (z, y, x)


    @mo.persistent_cache
    def render_space_stops(rotation_deg, shift_mm):
        """Draw one 3D scan at each stop from voxel space to physical space.

        Returns one PNG and one voxel-to-world affine per stop.
        """
        root = Path(
            cf.datasets.fetch_nunez_elizalde_2022(
                subjects="CR022",
                sessions="20201007",
                datatypes="susi",
                print_citation=False,
            )
        )
        scan = (
            cf.load(root / "sub-CR022/ses-20201007/susi/sub-CR022_ses-20201007_pwd.nii.gz")
            .fusi.scale.db()
            .compute()
        )

        # Put the first voxel at the world origin, so the affine starts as a pure zoom.
        zoomed = scan.fusi.translate([-scan.fusi.origin[dim] for dim in "zyx"])
        spacing = np.array([scan.fusi.spacing[dim] for dim in "kji"])
        # Voxel space has no physical size: draw every voxel as a cube of the smallest
        # spacing, so the scan keeps a comparable size on screen.
        unit = spacing.min()
        voxels = zoomed.fusi.zoom(unit / spacing, center=(0, 0, 0))
        rotated = zoomed.fusi.rotate(rotation_deg, "z", center=(0, 0, 0))
        translated = rotated.fusi.translate(shift_mm)
        stops = [voxels, zoomed, rotated, translated]
        affines = [np.eye(4)] + [stop.fusi.affine.voxel_to_world for stop in stops[1:]]

        # One fixed frame for all stops, so the scan visibly moves inside it.
        boxes = [stop.fusi.affine.bounding_box for stop in stops]
        limits = {
            dim: (
                min(box[dim][0] for box in boxes) - 0.6,
                max(box[dim][1] for box in boxes) + 0.6,
            )
            for dim in "zyx"
        }
        box_before = zoomed.fusi.affine.bounding_box
        # (slice normal, horizontal axis, vertical axis, name)
        views = [
            ("z", "x", "y", "coronal"),
            ("x", "z", "y", "sagittal"),
            ("y", "x", "z", "horizontal"),
        ]
        voxel_dim = dict(zip("zyx", "kji"))

        def render(step):
            data = stops[step]
            center = data.fusi.affine.geometry_center
            width_x, height_y, width_z = (np.ptp(limits[dim]) for dim in "xyz")
            fig = plt.figure(
                figsize=(11, 11 * (height_y + width_z) / (width_x + width_z))
            )
            grid = fig.add_gridspec(
                2,
                2,
                width_ratios=[width_x, width_z],
                height_ratios=[height_y, width_z],
                wspace=0.03,
                hspace=0.03,
                left=0.075,
                right=0.995,
                top=0.995,
                bottom=0.075,
            )
            axes = {
                "coronal": fig.add_subplot(grid[0, 0]),
                "sagittal": fig.add_subplot(grid[0, 1]),
                "horizontal": fig.add_subplot(grid[1, 0]),
            }
            for normal, horizontal, vertical, name in views:
                ax = axes[name]
                # plot_volume resamples the scan onto the screen grid through its affine.
                cf.plotting.plot_volume(
                    data,
                    slice_mode=normal,
                    slice_coords=[center[normal]],
                    axes=ax,
                    transpose=name == "sagittal",
                    cmap="gray",
                    vmin=-16,
                    vmax=-6,
                    show_colorbar=False,
                    show_titles=False,
                    bg_color=PANEL,
                    resample_fill_value=np.nan,
                )
                ax.set_xlim(*limits[horizontal])
                ax.set_ylim(*limits[vertical][::-1])
                ax.set_aspect("equal")
                for axis, dim in ((ax.xaxis, horizontal), (ax.yaxis, vertical)):
                    if step == 0:
                        indices = np.arange(0, limits[dim][1] / unit, 40)
                        axis.set_ticks(indices * unit, [f"{index:g}" for index in indices])
                        axis.set_label_text(f"{voxel_dim[dim]} (voxels)")
                    else:
                        axis.set_ticks(np.arange(0, limits[dim][1], 2))
                        axis.set_label_text(f"{dim} (mm)")
                ax.grid(True, color="white", alpha=0.13, lw=0.7)
                ax.plot(0, 0, "+", color=ACCENT, ms=20, mew=3, zorder=5)
                if step >= 2:
                    (h0, h1), (v0, v1) = box_before[horizontal], box_before[vertical]
                    ax.add_patch(
                        Rectangle(
                            (h0, v0),
                            h1 - h0,
                            v1 - v0,
                            fill=False,
                            edgecolor=OUTLINE,
                            linestyle="--",
                            linewidth=1.8,
                            zorder=4,
                        )
                    )
                ax.text(
                    0.985,
                    0.975,
                    name,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    alpha=0.75,
                )
            # The coronal view shares x with the view below it and y with the one beside it.
            axes["coronal"].tick_params(labelbottom=False)
            axes["coronal"].set_xlabel("")
            axes["sagittal"].tick_params(labelleft=False)
            axes["sagittal"].set_ylabel("")

            handles = [
                Line2D(
                    [], [], marker="+", ls="", color=ACCENT, ms=16, mew=3,
                    label="origin (0, 0, 0)",
                )
            ]
            if step >= 2:
                handles.append(
                    Line2D([], [], ls="--", color=OUTLINE, lw=1.8, label="before rotation")
                )
            fig.legend(
                handles=handles,
                loc="center",
                bbox_to_anchor=grid[1, 1].get_position(fig),
                frameon=False,
            )
            return fig_png(fig)

        return [render(step) for step in range(len(stops))], affines


    space_pngs, space_affines = render_space_stops(ROTATION_DEG, SHIFT_MM)
    return ROTATION_DEG, SHIFT_MM, space_affines, space_pngs


@app.cell(hide_code=True)
def _(ROTATION_DEG, SHIFT_MM, affine_latex, mo, np, space_affines, space_pngs):
    def space_stop(step):
        """Build the picture and text for one stop between voxel space and physical space."""
        names = ["Voxels", "+ zoom", "+ rotation", "+ translation"]
        voxel_size = " × ".join(f"{size:.2g}" for size in np.diag(space_affines[1])[:3])
        captions = [
            "The axes are voxel indices. The brain looks compressed.",
            f"The voxel size is {voxel_size} mm (z, y, x). The brain has its true shape.",
            f"We rotate the volume {abs(ROTATION_DEG):g}° about the z axis.",
            "We translate the volume by ({:g}, {:g}, {:g}) mm along (z, y, x).".format(
                *SHIFT_MM
            ),
        ]
        trail = " → ".join(
            f"**{name}**" if index == step else f"<span style='opacity: 0.45'>{name}</span>"
            for index, name in enumerate(names)
        )

        # The entries that changed at this stop are highlighted.
        equation = (
            r"$$\begin{bmatrix} z \\ y \\ x \\ 1 \end{bmatrix} = \underbrace{"
            + affine_latex(space_affines[step], space_affines[max(step - 1, 0)])
            + r"}_{\text{affine}} \begin{bmatrix} k \\ j \\ i \\ 1 \end{bmatrix}$$"
        )

        # The text block keeps one height at the four stops, so nothing moves between
        # them: the caption reserves two lines and the last note reserves its space.
        last = step == len(names) - 1
        text = [
            mo.md(f"<span style='white-space: nowrap; font-size: 0.8em'>{trail}</span>"),
            mo.md(captions[step]).style({"min-height": "4em"}),
            mo.md(equation),
            mo.md(
                "The voxel values do not change. Only the affine changes. "
                "To show each image, we **resample** the volume."
            ).style({"visibility": "visible" if last else "hidden"}),
        ]
        return mo.hstack(
            [mo.image(space_pngs[step], width="100%"), mo.vstack(text, gap=1.5)],
            widths=[1.3, 1],
            gap=2,
            align="center",
        )

    return (space_stop,)


@app.cell(hide_code=True)
def _(PANEL, Path, cf, fig_png, mo, np, plt):
    @mo.persistent_cache
    def render_session_registration():
        """Register one rat slice from two consecutive days.

        Returns the overlays as PNGs (before, after, and the pair), and the affines of
        the registration.
        """
        sessions = ["20220523", "20220524"]
        root = Path(
            cf.datasets.fetch_cybis_pereira_2026(
                datasets="rawdata",
                subjects="rat75",
                sessions=sessions,
                acqs="slice32",
                datatypes="susi",
                print_citation=False,
            )
        )

        def load_angio(session):
            angio = (
                cf.load(
                    root
                    / f"sub-rat75/ses-{session}/susi"
                    / f"sub-rat75_ses-{session}_acq-slice32_rec-minframe2d_pwd.nii.gz"
                )
                .fusi.scale.db()
                .compute()
            )
            # This dataset's sform space is not metric: work in its qform space.
            return angio.fusi.affine.apply(angio.affines["world_to_qform"])

        fixed, moving = load_angio(sessions[0]), load_angio(sessions[1])
        # register_volume returns the transform from fixed world to moving world.
        registered, fixed_to_moving, _ = cf.registration.register_volume(
            moving=moving, fixed=fixed, transform_type="rigid"
        )
        affines = {
            "fixed": fixed.fusi.affine.voxel_to_world,
            "moving": moving.fusi.affine.voxel_to_world,
            "moving_to_fixed": np.linalg.inv(fixed_to_moving),
        }

        def overlay(panels):
            fig, axes = plt.subplots(
                1,
                len(panels),
                figsize=(6.5 * len(panels), 5.2),
                layout="constrained",
                squeeze=False,
            )
            for ax, (title, image) in zip(axes[0], panels):
                cf.plotting.plot_composite(
                    fixed,
                    image,
                    axes=ax,
                    show_titles=False,
                    show_axes=False,
                    bg_color=PANEL,
                )
                ax.set_title(title)
            return fig_png(fig)

        before = ("Before", moving)
        after = ("After rigid registration", registered)
        pngs = {
            "before": overlay([before]),
            "after": overlay([after]),
            "pair": overlay([before, after]),
        }
        return pngs, affines


    sessions_pngs, sessions_affines = render_session_registration()
    return sessions_affines, sessions_pngs


@app.cell(hide_code=True)
def _(FuncAnimation, Path, PillowWriter, cf, mo, np, plt, tempfile):
    @mo.persistent_cache
    def render_motion_correction():
        """Animate a freely-moving recording, before and after motion correction.

        Returns a looping animated WebP.
        """
        root = Path(
            cf.datasets.fetch_cybis_pereira_2026(
                datasets="rawdata",
                subjects="rat75",
                sessions="20220523",
                acqs="slice32",
                datatypes="fusi",
                print_citation=False,
            )
        )
        # A short window with visible motion.
        recording = (
            cf.load(
                root
                / "sub-rat75/ses-20220523/fusi"
                / "sub-rat75_ses-20220523_task-openfield_acq-slice32_pwd.nii.gz"
            )
            .isel(time=slice(220, 340))
            .compute()
        )
        corrected = cf.registration.register_volumewise(
            recording,
            transform="rigid",
            moving_intensity_scaling="db",
            metric="correlation",
            learning_rate=1.0,
            show_progress=False,
        )

        # The motion is smaller than one voxel, so the full view hides it: zoom on a
        # column of cortex with its penetrating vessels.
        zoom = {"k": 0, "j": slice(4, 84), "i": slice(52, 98)}
        before = recording.fusi.scale.db().isel(zoom).values
        after = corrected.fusi.scale.db().isel(zoom).values
        vmin, vmax = np.nanpercentile(before, [2, 99.8])
        spacing = recording.fusi.spacing

        fig, axes = plt.subplots(1, 2, figsize=(8, 6.2), layout="constrained")
        images = []
        for ax, title, frames in zip(axes, ("Before", "After"), (before, after)):
            images.append(
                ax.imshow(
                    frames[0],
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                    aspect=spacing["j"] / spacing["i"],
                    interpolation="bicubic",
                )
            )
            ax.set_title(title)
            ax.set_axis_off()

        def show_frame(index):
            images[0].set_data(before[index])
            images[1].set_data(after[index])

        # The recording runs at 2.5 frames per second: play it about 5 times faster.
        animation = FuncAnimation(fig, show_frame, frames=before.shape[0])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "motion.webp"
            animation.save(path, writer=PillowWriter(fps=12))
            data = path.read_bytes()
        plt.close(fig)
        return data


    motion_animation = render_motion_correction()
    return (motion_animation,)


@app.cell(hide_code=True)
def _(PANEL, Path, cf, fig_png, mo, np, plt):
    @mo.persistent_cache
    def render_atlas_overlay():
        """Register one mouse slice to a fUSI template and draw the Allen atlas on it."""
        template = cf.datasets.fetch_template_pepe_mariani_2026(
            print_citation=False
        ).compute()
        root = Path(
            cf.datasets.fetch_nunez_elizalde_2022(
                subjects="CR022",
                sessions="20201007",
                tasks="spontaneous",
                acqs="slice02",
                datatypes="fusi",
                print_citation=False,
            )
        )
        recording = (
            cf.load(
                root
                / "sub-CR022/ses-20201007/fusi"
                / "sub-CR022_ses-20201007_task-spontaneous_acq-slice02_pwd.nii.gz"
            )
            .mean(dim="time")
            .fusi.scale.db()
            .compute()
        )

        # Rough manual placement of the slice in the template (done once in napari). The
        # registration needs it as a starting point and then refines it.
        placement = np.eye(4)
        placement[:3, 3] = (5.594638656430411, -2.50293925701927, 5.6650243788545875)
        slice_z = placement[0, 3] + recording.fusi.origin["z"]
        fixed = template.sel(z=slice(slice_z - 1.0, slice_z + 1.0)).fusi.scale.db()
        _, affine, _ = cf.registration.register_volume(
            moving=recording,
            fixed=fixed,
            transform_type="affine",
            metric="correlation",
            convergence_window_size=50,
            number_of_iterations=500,
            learning_rate=1.0,
            initialization=np.linalg.inv(placement),
        )

        # The template carries its own transform to Allen space: chain the two.
        recording_to_atlas = template.attrs["affines"]["world_to_sform"] @ np.linalg.inv(
            affine
        )
        atlas = cf.datasets.fetch_brainglobe_atlas("allen_mouse_100um", check_latest=False)
        labels = atlas.atlas.resample_like(recording, recording_to_atlas).annotation

        fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
        plotter = cf.plotting.plot_volume(
            recording,
            axes=ax,
            slice_mode="z",
            cmap="gray",
            show_colorbar=False,
            show_titles=False,
            show_axes=False,
            bg_color=PANEL,
        )
        plotter.add_contours(labels)
        return fig_png(fig)


    atlas_png = render_atlas_overlay()
    return (atlas_png,)


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
    # Data acquisition · Which route?

    | Route | What you get | Trade-off |
    | :-- | :-- | :-- |
    | **Iconeus One** | Integrated acquisition and analysis | Plug & play; less sequence control |
    | **Vevo F2xc + VADA** | General ultrasound research with configurable sequences and channel-data access | More acquisition control; fUSI needs validation |
    | **Verasonics + AUTC** | Research scanner + proprietary software and support | More configurable; vendor-specific software |
    | **Verasonics + EffUSIve** | Research scanner + open-source software and community support | Most flexible; alpha, requires expertise |

    **Choose by control, expertise, budget and experimental needs.**
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    def _card(photo, brand, model):
        image = mo.image(
            Path("assets") / f"{photo}.webp",
            alt=f"{brand} {model} scanner",
            width="100%",
            style={"display": "block", "margin": "0"},
        ).text
        return f"""
        <div style="min-width:0; padding:.5rem; border:1px solid #4b6269; border-radius:12px; background:#151a1c; text-align:center">
          <div style="width:100%; overflow:hidden">{image}</div>
          <div style="min-height:2.5em; margin-top:3px; line-height:1.05">
            <span style="color:#9fb7bc; font-size:.8em">{brand}</span><br><strong style="font-size:clamp(.75em, 1.5vw, 1em); overflow-wrap:anywhere">{model}</strong>
          </div>
        </div>"""

    cards = "".join(
        _card(*scanner) for scanner in (
            ("iconeus-one", "Iconeus", "Iconeus One"),
            ("verasonics-nxt", "Verasonics", "Vantage NXT"),
            ("visualsonics-vevo-f2xc", "VisualSonics", "Vevo F2xc"),
            ("us4us-us4r", "us4us", "us4R"),
            ("opensonics-openexplorer", "OpenSonics", "OpenExplorer"),
        )
    )
    mo.md(f"""
    # Data acquisition · The scanners

    <div style="display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.5rem; margin:1.5rem 0">{cards}</div>

    A scanner is only **one part** of a fUSI setup: check probe compatibility, computing / fast storage, and the complete acquisition software pipeline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Data acquisition · The software

    <div style="display:flex; align-items:stretch; gap:.6rem; margin:1.5rem 0">
      <div style="flex:1; min-width:0; padding:1rem; border:2px solid #4dd0e1; border-radius:12px"><strong>① Sequence</strong><br>Transmit &amp; receive</div>
      <div style="align-self:center; color:#ffd43b">→</div>
      <div style="flex:1; min-width:0; padding:1rem; border:2px solid #4dd0e1; border-radius:12px"><strong>② Reconstruct</strong><br>Beamform + clutter filter</div>
      <div style="align-self:center; color:#ffd43b">→</div>
      <div style="flex:1; min-width:0; padding:1rem; border:2px solid #4dd0e1; border-radius:12px"><strong>③ Monitor</strong><br>Live visualization</div>
      <div style="align-self:center; color:#ffd43b">→</div>
      <div style="flex:1; min-width:0; padding:1rem; border:2px solid #4dd0e1; border-radius:12px"><strong>④ Record</strong><br>Save + synchronize</div>
    </div>

    | Scanner | Software route | fUSI workflow |
    | :-- | :-- | :--: |
    | Iconeus One | Iconeus, bundled | <span style="color:#3ad9a4">✓</span> |
    | Verasonics | AUTC (proprietary) / EffUSIve (open-source, alpha) | <span style="color:#3ad9a4">✓</span> |
    | VisualSonics Vevo F2xc | VisualSonics | <span style="color:#ffd43b">? Verify</span> |
    | us4us us4R | Your own / third-party software | <span style="color:#ffd43b">? Verify</span> |
    | OpenSonics OpenExplorer | Your own / third-party software | <span style="color:#ffd43b">? Verify</span> |
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Standardization

    - Data Standardization (fUSI-BIDS)
      - Nice logo
      - Research Data Management (RDM)
        - Help your future self, your colleague, your student, your PI, the PhD student after you
      - fUSI-BIDS
      - Soft reminder to share data
    - Code Standardization (ConfUSIus)
      - Mainly what we will talk about in the following slides
      - Reproducibility crisis in neuroimaging etc etc
      - Improvement of methods as a community depends on methods sharing
      - Soft reminder to share code
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("# Voxel space → physical space"),
            mo.md("Some intro to physical space")
        ]
    )


    return


@app.cell(hide_code=True)
def _(fragment, mo, space_stop):
    # The four stops share one place on the slide. Each key press swaps to the next
    # stop, with no slide transition.
    _last = 3
    _layers = [
        fragment(
            space_stop(step),
            max(step - 1, 0),
            # The first stop shows until the first key press. The last stop stays.
            "fade-out" if step == 0 else "" if step == _last else "current-visible",
            "grid-area: 1 / 1; transition: none",
        ).text
        for step in range(_last + 1)
    ]

    mo.vstack(
        [
            mo.md("# Voxel space → physical space"),
            mo.Html(f'<div style="display: grid">{"".join(_layers)}</div>'),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(
    FIXED,
    MOVING,
    affine_latex,
    fragment,
    mo,
    sessions_affines,
    sessions_pngs,
):
    _fixed, _moving, _moving_to_fixed = (
        sessions_affines[name] for name in ("fixed", "moving", "moving_to_fixed")
    )
    # Sizes that keep the four matrices inside the slide on a small screen.
    _image_height = "28vh"
    _math_size = r"\scriptsize "


    def _labeled(affine, label, color=None, previous=None):
        """LaTeX of an affine with its name below it, in the color of its image."""
        name = r"\text{" + label + "}"
        if color:
            name = rf"\textcolor{{{color}}}{{{name}}}"
        return r"\underbrace{" + affine_latex(affine, previous) + "}_{" + name + "}"


    def _math(latex):
        return mo.md("$$" + _math_size + latex + "$$")


    # Key press 1 shows the state before registration: the overlay and the two affines.
    # Key press 2 shows the state after: the overlay, the transform and the new affine.
    # The moving image gets a new voxel-to-world affine: no resampling is necessary.
    mo.vstack(
        [
            mo.md("# Registration finds the transform"),
            mo.hstack(
                [
                    fragment(mo.image(sessions_pngs["before"], height=_image_height), 0),
                    fragment(mo.image(sessions_pngs["after"], height=_image_height), 1),
                ],
                justify="center",
            ),
            fragment(
                mo.md(
                    "Red is the fixed image. Cyan is the moving image, from the same "
                    "rat one day later."
                ),
                0,
            ),
            # The fixed affine stands apart from the product on its right.
            mo.hstack(
                [
                    fragment(
                        _math(_labeled(_fixed, "fixed: voxel-to-world", FIXED)),
                        0,
                        style="margin-right: 2rem",
                    ),
                    fragment(
                        _math(_labeled(_moving_to_fixed, "rigid transform") + r" \times"),
                        1,
                    ),
                    fragment(
                        _math(_labeled(_moving, "moving: voxel-to-world", MOVING)), 0
                    ),
                    fragment(
                        _math(
                            "= "
                            + _labeled(
                                _moving_to_fixed @ _moving,
                                "moving: new voxel-to-world",
                                MOVING,
                                _moving,
                            )
                        ),
                        1,
                    ),
                ],
                justify="center",
                gap=0.5,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(atlas_png, fragment, mo, motion_animation, sessions_pngs):
    # One image box size for the three columns, whatever the figure proportions.
    _box = {"width": "100%", "height": "300px", "object-fit": "contain"}


    def _part(index, title, png, text):
        """One column, shown at key press `index`."""
        column = mo.callout(
            mo.vstack([mo.md(f"## {title}"), mo.image(png, style=_box), mo.md(text)])
        )
        return fragment(column, index)


    mo.vstack(
        [
            mo.md("# Registration in a fUSI project"),
            mo.hstack(
                [
                    _part(
                        0,
                        "Motion correction",
                        motion_animation,
                        "We register each frame to a reference volume.",
                    ),
                    _part(
                        1,
                        "Across sessions and animals",
                        sessions_pngs["pair"],
                        "The same coordinates must show the same region.",
                    ),
                    _part(
                        2,
                        "Template → atlas",
                        atlas_png,
                        "An atlas-aligned template gives the labels.",
                    ),
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
    mo.md(r"""
    # Hands-on
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # fUSI-BIDS

    On top of good RDM, there are useful tools.

    - `pybids` in a dataset. Data selection.

    Maybe show that a standardized orientation helps with registration?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Title

    - item 1
    - item 2
    """)
    return


if __name__ == "__main__":
    app.run()
