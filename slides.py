import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="PUW 2026 — fUSI analysis & open-source tools",
    layout_file="layouts/slides.slides.json",
    css_file="theme.css",
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
    import SimpleITK as sitk
    import xarray as xr
    from brainglobe_atlasapi.structure_class import StructuresDict
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import binary_erosion, gaussian_filter

    return (
        FuncAnimation,
        Image,
        ImageDraw,
        ImageFont,
        Line2D,
        Path,
        PillowWriter,
        Rectangle,
        StructuresDict,
        binary_erosion,
        cf,
        gaussian_filter,
        io,
        mo,
        np,
        plt,
        sitk,
        tempfile,
        xr,
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
def _(fragment, mo):
    def swap_stops(stops, start=0):
        """Show stops one at a time, in one place on the slide.

        Each key press swaps to the next stop, with no slide transition. The first swap is
        at key press `start` (from 0): a stop can show its own fragments before that.
        """
        last = len(stops) - 1
        layers = [
            fragment(
                stop,
                start + max(step - 1, 0),
                # The first stop shows until the first swap. The last stop stays.
                "fade-out" if step == 0 else "" if step == last else "current-visible",
                "grid-area: 1 / 1; transition: none",
            ).text
            for step, stop in enumerate(stops)
        ]
        return mo.Html(f'<div style="display: grid">{"".join(layers)}</div>')


    def row(items, justify="center", gap=3):
        """Put items side by side, each at its own width, centered on one line.

        `justify` is a CSS `justify-content` value, and `gap` is in rem.
        """
        items = "".join(f"<div>{item.text}</div>" for item in items)
        style = f"display: flex; justify-content: {justify}; align-items: center"
        return mo.Html(f'<div style="{style}; gap: {gap}rem">{items}</div>')


    def captioned(picture, caption, height="50cqh"):
        """A picture or a film, above a short caption."""
        return mo.vstack(
            [
                mo.image(picture, height=height),
                mo.md(f"<span style='font-size: 0.8em'>{caption}</span>"),
            ],
            align="center",
            gap=0.5,
        )

    return captioned, row, swap_stops


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
        """Build the picture and text for one stop between voxel space and world space."""
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

    return


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
    return (sessions_pngs,)


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
def _(Path, cf, gaussian_filter, mo, np):
    # A synthetic power Doppler dataset: a wireframe Eiffel tower, scanned six times.
    # The first scan is the reference. Each other scan has one change: tune the values
    # here and the slides follow.
    TOWER_CHANGES = {
        "reference": ("none", 0),
        "translated": ("shift", 1.0),  # mm to the right
        "rotated": ("rotation", 10),  # lean to the right about the foot, degrees
        "sheared": ("shear", 0.18),  # mm to the right, per mm of height
        "bent": ("bend", 0.8),  # mm to the right, near the middle of the tower
        "other": ("other", (0.85, 7)),  # another tower: (size, lean in degrees)
    }
    TOWER_SHAPE = (20, 80, 20)  # voxels (k, j, i)
    TOWER_SPACING = (0.3, 0.15, 0.3)  # mm (z, y, x)
    TOWER_GROUND = 11.0  # depth of the foot of the tower in a scan, mm
    TOWER_MM_PER_M = 9.5 / 330  # the 330 m tower is 9.5 mm tall
    TOWER_FLOORS = (57, 115, 276)  # heights of the three floors, m


    def tower_half_width(height):
        """Half the width of the tower at a height. Both are in metres."""
        return 4 + 58.5 * np.exp(-height / 81)


    def tower_leg_width(height):
        """Width of one leg at a height below the second floor. Both are in metres."""
        return 25 - 10 * height / 115


    def tower_arch(across):
        """Height of the arch between two legs, at a distance from the axis, in metres."""
        return 50 - 30 * (across / 36) ** 2


    def tower_blood(change, amount, rng):
        """Blood in a wireframe Eiffel tower.

        `change` moves the tower: see `TOWER_CHANGES`. Returns the positions of the blood
        in a scan, in mm as (z, y, x), and the amount of blood at each position.
        """
        # (start, end, brightness) of each beam. Positions are (across, height, along)
        # in metres.
        beams = []

        def square(center, half, height):
            """Four corners of a horizontal square, in order around it."""
            return np.array(
                [
                    (center[0] + a * half, height, center[1] + b * half)
                    for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
                ]
            )

        def lattice(corners_at, levels):
            """A pylon: four columns, with a ring and cross-bracing at each level."""
            for low, high in zip(levels[:-1], levels[1:]):
                bottom, top = corners_at(low), corners_at(high)
                for n in range(4):
                    m = (n + 1) % 4
                    beams.append((bottom[n], top[n], 1.0))
                    beams.append((top[n], top[m], 0.6))
                    beams.append((bottom[n], top[m], 0.35))
                    beams.append((bottom[m], top[n], 0.35))

        def leg(side_x, side_z):
            def corners_at(height):
                width = tower_leg_width(height)
                center = tower_half_width(height) - width / 2
                return square((side_x * center, side_z * center), width / 2, height)

            return corners_at

        # Four separate legs up to the second floor, then one pylon to the top.
        for side_x in (-1, 1):
            for side_z in (-1, 1):
                lattice(leg(side_x, side_z), [0, 14, 28, 42, 57, 72, 86, 100, 115])
        lattice(
            lambda height: square((0, 0), tower_half_width(height), height),
            np.linspace(115, 300, 15),
        )

        # The three floors: bright square decks, a little wider than the tower.
        for height in TOWER_FLOORS:
            for half in np.linspace(0.25, 1, 4) * (tower_half_width(height) + 3):
                deck = square((0, 0), half, height)
                beams.extend((deck[n], deck[(n + 1) % 4], 1.5) for n in range(4))

        # The four arches between the legs, below the first floor.
        across = np.linspace(-1, 1, 25) * 36
        rise = tower_arch(across)
        for side in (-1, 1):
            arch = np.stack([across, rise, side * tower_half_width(rise)], axis=1)
            for face in (arch, arch[:, ::-1]):
                beams.extend((face[n], face[n + 1], 0.8) for n in range(len(face) - 1))

        # The spire.
        tip = np.array((0.0, 312.0, 0.0))
        beams.extend(
            (corner, tip, 1.0) for corner in square((0, 0), tower_half_width(300), 300)
        )
        beams.append((tip, np.array((0.0, 330.0, 0.0)), 1.0))

        # Blood: random points along the beams, in mm as (z, y, x). Depth is y, so the
        # tower stands on its foot in the image.
        starts, ends, brightness = (np.array(column, float) for column in zip(*beams))
        lengths = np.linalg.norm(ends - starts, axis=1) * TOWER_MM_PER_M
        counts = np.ceil(lengths / 0.01).astype(int)
        beam = np.repeat(np.arange(len(beams)), counts)
        position = starts[beam] + rng.random((beam.size, 1)) * (ends - starts)[beam]
        across, height, along = position.T * TOWER_MM_PER_M
        points = np.stack([along, TOWER_GROUND - height, across], axis=1)
        blood = (brightness * lengths / counts)[beam]
        foot = (0, TOWER_GROUND, 0)

        def lean(degrees):
            """Lean to the right about the foot. It stays in the (y, x) plane."""
            cos, sin = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
            points[:] -= foot
            points[:, 1:] = points[:, 1:] @ [[cos, -sin], [sin, cos]]
            points[:] += foot

        # Move the tower.
        if change == "shift":
            points[:, 2] += amount
        elif change == "rotation":
            lean(amount)
        elif change == "shear":
            points[:, 2] += amount * (TOWER_GROUND - points[:, 1])
        elif change == "bend":
            # A smooth bump with a radius of about 1.5 mm.
            distance = np.linalg.norm(points - (0, 5.5, 0), axis=1)
            points[:, 2] += amount * np.exp(-(distance**2) / (2 * 1.5**2))
        elif change == "other":
            size, degrees = amount
            points[:] = foot + size * (points - foot)
            lean(degrees)
        return points, blood


    @mo.persistent_cache
    def simulate_tower(change, amount, seed):
        """Simulate a power Doppler volume of a wireframe Eiffel tower.

        `change` moves the tower before the scan: see `TOWER_CHANGES`. Returns the power
        as a (k, j, i) array.
        """
        rng = np.random.default_rng(seed)
        points, blood = tower_blood(change, amount, rng)

        # The scan. In each frame, the probe sees random blood scatterers through its
        # point spread function. Power Doppler is the mean power of the frames.
        edges = [
            (np.arange(size + 1) - size / 2) * step
            for size, step in zip(TOWER_SHAPE, TOWER_SPACING)
        ]
        edges[1] -= edges[1][0]  # depth starts at the probe
        amplitude = np.sqrt(np.histogramdd(points, bins=edges, weights=blood)[0])
        psf = np.divide((0.15, 0.045, 0.06), TOWER_SPACING)  # standard deviation, voxels
        n_frames = 16
        signal = np.zeros(TOWER_SHAPE)
        for _ in range(n_frames):
            scatterers = rng.normal(size=TOWER_SHAPE) + 1j * rng.normal(size=TOWER_SHAPE)
            signal += np.abs(gaussian_filter(amplitude * scatterers, psf)) ** 2
        # The noise grows with depth, like the gain: 0.6 dB per mm. The brightest blood
        # is 30 dB above the noise at the probe.
        depth = edges[1][:-1, None]
        noise = 10 ** (0.06 * depth) * rng.gamma(n_frames, 1 / n_frames, TOWER_SHAPE)
        return (1000 * signal / signal.max() + noise).astype(np.float32)


    def project_tower(volume):
        """The maximum intensity projection along z, in the place of the middle slice."""
        middle = volume.isel(k=[volume.sizes["k"] // 2])
        return middle.copy(data=volume.max("k").values[None])


    # Save each scan as NIfTI, then load it like any other power Doppler file.
    towers = {}
    for _seed, (_name, (_change, _amount)) in enumerate(TOWER_CHANGES.items()):
        _path = Path("data/eiffel") / f"sub-eiffel_ses-{_name}_pwd.nii"
        _path.parent.mkdir(parents=True, exist_ok=True)
        cf.save(
            cf.create_voxeldata(
                simulate_tower(_change, _amount, _seed),
                dims=("k", "j", "i"),
                spacing=TOWER_SPACING,
                attrs={
                    "Description": "Simulated power Doppler of a wireframe Eiffel tower."
                },
            ),
            _path,
        )
        towers[_name] = cf.load(_path).compute()

    # The slides show and register one picture of each scan: its projection along z.
    tower_mips = {name: project_tower(tower) for name, tower in towers.items()}
    return (
        TOWER_FLOORS,
        TOWER_GROUND,
        TOWER_MM_PER_M,
        project_tower,
        tower_arch,
        tower_blood,
        tower_half_width,
        tower_leg_width,
        tower_mips,
    )


@app.cell(hide_code=True)
def _(ACCENT, PANEL, fig_png, mo, np, plt, tower_mips):
    @mo.persistent_cache
    def render_tower_spaces(scan):
        """Draw the projection of the tower in voxel space, then in world space.

        Returns one PNG per space.
        """
        power_db = 10 * np.log10(scan.values[0] / scan.values.max())
        # Voxel space has no size in mm: draw every voxel as a square of the smallest
        # spacing, so the two spaces fit in one frame.
        unit = min(scan.fusi.spacing[dim] for dim in "ji")
        voxel_dim = {"x": "i", "y": "j"}
        voxel_box = {dim: (0, scan.sizes[voxel_dim[dim]] * unit) for dim in "xy"}
        world_box = scan.fusi.affine.bounding_box
        limits = {
            dim: (min(world_box[dim][0], 0) - 0.4, world_box[dim][1] + 0.4) for dim in "xy"
        }

        def render(box, world):
            fig, ax = plt.subplots(figsize=(4.4, 7.6), layout="constrained")
            ax.imshow(
                power_db,
                cmap="gray",
                vmin=-27,
                vmax=0,
                extent=(*box["x"], *box["y"][::-1]),
                interpolation="bicubic",
            )
            ax.set_facecolor(PANEL)
            ax.set_xlim(*limits["x"])
            ax.set_ylim(*limits["y"][::-1])
            for axis, dim in ((ax.xaxis, "x"), (ax.yaxis, "y")):
                if world:
                    axis.set_major_locator(plt.MultipleLocator(2))
                    axis.set_label_text(f"{dim} (mm)")
                else:
                    indices = np.arange(0, scan.sizes[voxel_dim[dim]] + 1, 10)
                    axis.set_ticks(indices * unit, indices)
                    axis.set_label_text(f"{voxel_dim[dim]} (voxels)")
            ax.grid(True, color="white", alpha=0.13, lw=0.7)
            ax.plot(0, 0, "+", color=ACCENT, ms=20, mew=3, zorder=5, label="origin")
            fig.legend(loc="outside lower center", frameon=False)
            return fig_png(fig)

        return render(voxel_box, world=False), render(world_box, world=True)


    tower_space_pngs = render_tower_spaces(tower_mips["reference"])
    return (tower_space_pngs,)


@app.cell(hide_code=True)
def _(Image, cf, fig_png, io, mo, np, plt, sitk, tower_mips):
    def loop_film(pictures):
        """Make a looping film (an animated WebP) from the pictures of a search.

        The film keeps at most 36 pictures. It shows the first one for 0.5 seconds, the
        search for 1 second, and the last one for 1 second.
        """
        steps = np.unique(np.linspace(0, len(pictures) - 1, 36).round().astype(int))
        between = max(len(steps) - 2, 1)
        film = io.BytesIO()
        pictures[0].save(
            film,
            format="WEBP",
            save_all=True,
            append_images=[pictures[step] for step in steps[1:]],
            duration=[500] + [round(1000 / between)] * (len(steps) - 2) + [1000],
            loop=0,
        )
        return film.getvalue()


    @mo.persistent_cache
    def film_tower_registration(fixed, moving, **settings):
        """Register the projection of a moving tower to the reference, and film it.

        Returns the picture before registration (PNG), the film (a looping animated
        WebP), and the transform that ConfUSIus found.
        """
        peak = float(fixed.max())
        # 40 pixels for each mm, as (width, height).
        size = [round(40 * fixed.sizes[dim] * fixed.fusi.spacing[dim]) for dim in "ij"]

        def shade(scan):
            """Shades from 0 to 1, over the top 22 dB of the reference."""
            return np.clip(1 + 10 * np.log10(scan[0] / peak) / 22, 0, 1)

        reference = shade(fixed.values)

        def picture(scan):
            """The reference in red and the moving scan in cyan: they add to white."""
            moved = shade(scan)
            colors = np.uint8(255 * np.dstack([reference, moved, moved]))
            return Image.fromarray(colors).resize(size, Image.Resampling.BICUBIC)

        # The state before registration, then one more at each iteration.
        pictures = [picture(moving.values)]

        class Recorder:
            """ConfUSIus calls `update` at each iteration of the optimizer."""

            def __init__(self, method, fixed_image, moving_image, *, resample_kwargs, **_):
                self.resample = lambda: sitk.Resample(
                    moving_image,
                    fixed_image,
                    method.GetInitialTransform(),
                    sitk.sitkLinear,
                    resample_kwargs["fill_value"],
                )

            def update(self):
                # `.T` gives back the (k, j, i) order.
                pictures.append(picture(sitk.GetArrayFromImage(self.resample()).T))

            def close(self):
                pass

        _, transform, _ = cf.registration.register_volume(
            moving,
            fixed,
            fixed_intensity_scaling="sqrt",
            moving_intensity_scaling="sqrt",
            show_progress=True,
            progress_plotter=Recorder,
            **settings,
        )

        before = io.BytesIO()
        pictures[0].save(before, format="PNG")
        return before.getvalue(), loop_film(pictures), transform


    @mo.persistent_cache
    def render_tower_warp(moving, bspline):
        """Draw the warp of a B-spline registration, as arrows on the moving tower (PNG).

        Each arrow shows the motion of one position of the moving tower, at its true
        length. The color of an arrow is its length.
        """
        # ConfUSIus samples the B-spline on the grid of the tower. This field goes from the
        # reference to the moving tower: its inverse is the motion of the moving tower.
        field = cf.registration.sample_displacement_field_like(bspline, moving)
        field = cf.registration.invert_displacement_field(field)
        # One arrow each 0.9 mm along x and each 0.6 mm along y: longer than any arrow.
        arrows = field.isel(k=0, j=slice(2, None, 4), i=slice(1, None, 3))
        u, v = arrows.sel(component="i").values, arrows.sel(component="j").values

        box = moving.fusi.affine.bounding_box
        # 40 pixels for each mm, like the films.
        fig = plt.figure(figsize=[0.4 * (box[dim][1] - box[dim][0]) for dim in "xy"])
        ax = fig.add_axes((0, 0, 1, 1))
        # Shades from 0 to 1, over the top 22 dB of the tower.
        power = moving.values[0] / moving.values.max()
        ax.imshow(
            np.clip(1 + 10 * np.log10(power) / 22, 0, 1),
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=(*box["x"], *box["y"][::-1]),
            interpolation="bicubic",
        )
        ax.quiver(
            arrows["x"].values,
            arrows["y"].values,
            u,
            v,
            np.hypot(u, v),
            cmap="autumn",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.012,
        )
        ax.axis("off")
        return fig_png(fig, dpi=100)


    # ponytail: one resolution is enough for these small changes on a coarse picture. If
    # a larger change stops the search too early, add `use_multi_resolution=True`.
    _linear = dict(learning_rate=1.0, number_of_iterations=300)
    _settings = {
        "translated": dict(transform_type="translation", **_linear),
        "rotated": dict(transform_type="rigid", **_linear),
        "sheared": dict(transform_type="affine", **_linear),
        # The mesh is (k, j, i): one cell across the plane. Keep the automatic step size
        # here: with a fixed `learning_rate`, this search diverges.
        "bent": dict(
            transform_type="bspline", mesh_size=(1, 8, 2), number_of_iterations=60
        ),
    }
    tower_registrations = {
        name: film_tower_registration(tower_mips["reference"], tower_mips[name], **settings)
        for name, settings in _settings.items()
    }
    tower_warp_png = render_tower_warp(tower_mips["bent"], tower_registrations["bent"][2])
    return loop_film, tower_registrations, tower_warp_png


@app.cell(hide_code=True)
def _(
    Path,
    StructuresDict,
    TOWER_FLOORS,
    TOWER_GROUND,
    TOWER_MM_PER_M,
    cf,
    gaussian_filter,
    np,
    project_tower,
    tower_arch,
    tower_blood,
    tower_half_width,
    tower_leg_width,
    xr,
):
    # The template of the tower and its atlas. The template has half the voxel size of a
    # scan. Its world origin is on the ground, at the outer edge of the left legs.
    TEMPLATE_SHAPE = (40, 160, 40)  # voxels (k, j, i)
    TEMPLATE_SPACING = (0.15, 0.075, 0.15)  # mm (z, y, x)
    TEMPLATE_START = (-3.0, -10.75, -1.2)  # mm (z, y, x): where the field of view starts
    # The regions of the atlas, from the ground up: (name, color).
    TOWER_REGIONS = {
        1: ("Floor", "#ff9f1c"),
        2: ("Platform 1", "#ffe066"),
        3: ("Platform 2", "#9be564"),
        4: ("Top", "#c79bff"),
    }


    def build_tower_atlas():
        """Build the template of the tower and its regions, as a ConfUSIus atlas."""
        foot = tower_half_width(0) * TOWER_MM_PER_M
        edges = [
            first + np.arange(size + 1) * step
            for first, size, step in zip(TEMPLATE_START, TEMPLATE_SHAPE, TEMPLATE_SPACING)
        ]

        # The template: what the mean of many scans gives. It has no speckle and no noise,
        # and it is a little smoother than one scan, because towers never align exactly.
        points, blood = tower_blood("none", 0, np.random.default_rng(100))
        points += (0, -TOWER_GROUND, foot)
        power = np.histogramdd(points, bins=edges, weights=blood)[0]
        power = gaussian_filter(power, np.divide((0.15, 0.045, 0.06), TEMPLATE_SPACING))
        template = (1000 * power / power.max() + 1).astype(np.float32)

        # The regions: the solid envelope of the tower, cut at the three floors. Positions
        # are (across, height, along) in metres, as in `tower_blood`.
        z, y, x = np.meshgrid(*[(e[:-1] + e[1:]) / 2 for e in edges], indexing="ij")
        across, height, along = np.array([x - foot, -y, z]) / TOWER_MM_PER_M
        margin = 4  # the envelope is a little wider than the beams
        half_width = tower_half_width(np.clip(height, 0, None))
        inside = (
            (height >= 0)
            & (height <= 330 + margin)
            & (abs(across) <= half_width + margin)
            & (abs(along) <= half_width + margin)
        )
        for distance in (across, along):
            # The opening between two legs, under an arch.
            between_legs = abs(distance) < half_width - tower_leg_width(height) - margin
            inside &= ~(between_legs & (height < tower_arch(distance) - margin))
        regions = (inside * (np.digitize(height, TOWER_FLOORS) + 1)).astype(np.int32)
        sides = np.where(across < 0, 1, 2).astype(np.int8)

        def volume(data, **attrs):
            return cf.create_voxeldata(
                data,
                dims=("k", "j", "i"),
                spacing=TEMPLATE_SPACING,
                origin=[e[0] + step / 2 for e, step in zip(edges, TEMPLATE_SPACING)],
                attrs=attrs,
            )

        # ConfUSIus reads the names and the colors of the regions from a BrainGlobe
        # structure tree. Its root is the whole tower.
        def structure(n, name, color, path):
            return {
                "id": n,
                "name": name,
                "acronym": name.lower(),
                "rgb_triplet": [int(color[c : c + 2], 16) for c in (1, 3, 5)],
                "structure_id_path": path,
                "mesh_filename": None,
            }

        structures = StructuresDict(
            [
                structure(5, "Tower", "#ffffff", [5]),
                *(structure(n, *region, [5, n]) for n, region in TOWER_REGIONS.items()),
            ]
        )
        return xr.Dataset(
            {
                "template": volume(template, cmap="gray"),
                "annotation": volume(
                    regions,
                    rgb_lookup={n: info["rgb_triplet"] for n, info in structures.items()},
                    roi_labels={n: info["name"] for n, info in structures.items()},
                ),
                "hemispheres": volume(sides, left=1, right=2),
            },
            attrs={
                "name": "eiffel_tower",
                "structures": structures,
                "world_to_base": np.eye(4),
            },
        )


    # Save the atlas, then load it like any other ConfUSIus atlas.
    _path = Path("data/eiffel/tpl-eiffel_atlas.zarr")
    cf.io.save_atlas(build_tower_atlas(), _path, mode="w")
    tower_atlas = cf.io.load_atlas(_path).compute()
    # The slides use its projection along z, like the scans.
    tower_atlas_mip = tower_atlas.map(project_tower, keep_attrs=True)
    return TOWER_REGIONS, tower_atlas_mip


@app.cell(hide_code=True)
def _(
    ACCENT,
    PANEL,
    TOWER_REGIONS,
    fig_png,
    gaussian_filter,
    mo,
    np,
    plt,
    tower_atlas_mip,
):
    @mo.persistent_cache
    def render_tower_atlas(template, regions):
        """Draw the template in its world space, under the regions of the atlas (PNG)."""
        box = template.fusi.affine.bounding_box
        fig, ax = plt.subplots(figsize=(4.4, 7.6), layout="constrained")
        ax.imshow(
            10 * np.log10(template.values[0] / template.values.max()),
            cmap="gray",
            vmin=-27,
            vmax=0,
            extent=(*box["x"], *box["y"][::-1]),
            interpolation="bicubic",
        )
        ax.set_facecolor(PANEL)
        ax.set_xlim(box["x"][0] - 0.4, box["x"][1] + 0.4)
        ax.set_ylim(box["y"][1] + 0.4, box["y"][0] - 0.4)
        for axis, dim in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            axis.set_major_locator(plt.MultipleLocator(2))
            axis.set_label_text(f"{dim} (mm)")
        ax.grid(True, color="white", alpha=0.13, lw=0.7)
        ax.plot(0, 0, "+", color=ACCENT, ms=20, mew=3, zorder=5, label="origin")
        # World positions of the voxel centers, along each axis of the picture.
        x, y = regions["x"].isel(k=0, j=0), regions["y"].isel(k=0, i=0)
        for label, (_, color) in TOWER_REGIONS.items():
            # A little blur takes the voxel steps out of the outline.
            smooth = gaussian_filter(np.float32(regions.values[0] == label), 0.8)
            ax.contour(x, y, smooth, levels=[0.5], colors=color, linewidths=2)
        fig.legend(loc="outside lower center", frameon=False)
        return fig_png(fig)


    tower_atlas_png = render_tower_atlas(
        tower_atlas_mip.atlas.template, tower_atlas_mip.atlas.annotation
    )
    return (tower_atlas_png,)


@app.cell(hide_code=True)
def _(
    Image,
    TOWER_REGIONS,
    binary_erosion,
    cf,
    gaussian_filter,
    io,
    loop_film,
    mo,
    np,
    sitk,
    tower_atlas_mip,
    tower_mips,
):
    @mo.persistent_cache
    def film_template_registration(template, regions, volume, **settings):
        """Register a volume to a template, and film it under the regions of the atlas.

        The film shows the template in red and the volume in cyan, in the space of the
        template. Returns the film (a looping animated WebP), and two pictures (PNG) of the
        registered volume in grey under the regions: in the space of the template, and in
        its own space. Also returns the affine from the template to the volume.
        """

        def shade(values, peak):
            """Shades from 0 to 1 for a (k, j, i) array, over the top 22 dB under a peak."""
            power = np.maximum(values[0], 1e-9) / float(peak)
            return np.clip(1 + 10 * np.log10(power) / 22, 0, 1)

        red, grey = (
            shade(template.values, template.max()),
            shade(volume.values, volume.max()),
        )

        def picture(red, cyan, grid, labels):
            """Shades in red and in cyan, under the outlines of the regions.

            The same shades in red and in cyan give a grey picture.
            """
            # 40 pixels for each mm, as (width, height).
            size = [round(40 * grid.sizes[dim] * grid.fusi.spacing[dim]) for dim in "ij"]
            image = Image.fromarray(np.uint8(255 * np.dstack([red, cyan, cyan])))
            image = np.array(image.resize(size, Image.Resampling.BICUBIC))
            for label, (_, color) in TOWER_REGIONS.items():
                # A little blur takes the voxel steps out of the edge of the region. Then
                # keep a line 2 pixels wide inside it.
                region = gaussian_filter(np.float32(labels[0] == label), 0.8)
                region = Image.fromarray(np.uint8(255 * region))
                region = np.array(region.resize(size, Image.Resampling.BICUBIC)) > 127
                image[region & ~binary_erosion(region, iterations=2)] = [
                    int(color[c : c + 2], 16) for c in (1, 3, 5)
                ]
            return Image.fromarray(image)

        def png(image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()

        # The pictures of the film, then the last state of the volume and of the regions.
        frames, moved, pulled = [], [], []

        class Recorder:
            """ConfUSIus calls `update` at each iteration of the optimizer."""

            def __init__(self, method, fixed_image, moving_image, *, resample_kwargs, **_):
                # The regions, as an image on the grid of the template. `.T` goes between
                # the (k, j, i) order and the order of SimpleITK.
                labels = np.broadcast_to(regions.values, fixed_image.GetSize()).T
                labels = sitk.GetImageFromArray(np.ascontiguousarray(labels, np.uint8))
                labels.CopyInformation(fixed_image)
                # The grid of the volume with 2 samples for each voxel of the picture: the
                # regions keep a fine outline in the space of the volume.
                fine = sitk.Expand(moving_image, [1, 2, 2])

                def resample():
                    # The transform goes from the template to the volume. It pulls the
                    # volume onto the template. Its inverse pulls the regions onto the
                    # volume.
                    transform = method.GetInitialTransform()
                    moved = sitk.Resample(
                        moving_image,
                        fixed_image,
                        transform,
                        sitk.sitkLinear,
                        resample_kwargs["fill_value"],
                    )
                    pulled[:] = [
                        sitk.Resample(
                            labels, fine, transform.GetInverse(), sitk.sitkNearestNeighbor
                        )
                    ]
                    return sitk.GetArrayFromImage(moved).T

                self.resample = resample
                # The state before the first iteration.
                self.update()

            def update(self):
                moved[:] = [shade(self.resample(), volume.max())]
                frames.append(picture(red, moved[0], template, regions.values))

            def close(self):
                pass

        _, affine, _ = cf.registration.register_volume(
            volume,
            template,
            fixed_intensity_scaling="sqrt",
            moving_intensity_scaling="sqrt",
            show_progress=True,
            progress_plotter=Recorder,
            **settings,
        )
        # The two results show the volume alone, in grey, under the regions.
        in_template = picture(moved[0], moved[0], template, regions.values)
        in_world = picture(grey, grey, volume, sitk.GetArrayFromImage(pulled[0]).T)
        return loop_film(frames), png(in_template), png(in_world), affine


    # The towers do not overlap at the start: their spaces have different origins. The
    # search starts with their centers of mass aligned.
    tower_template_registration = film_template_registration(
        tower_atlas_mip.atlas.template,
        tower_atlas_mip.atlas.annotation,
        tower_mips["other"],
        transform_type="affine",
        initialization="center_moments",
        learning_rate=1.0,
        number_of_iterations=300,
    )
    return (tower_template_registration,)


@app.cell(hide_code=True)
def _(
    Image,
    ImageDraw,
    ImageFont,
    PANEL,
    Rectangle,
    fig_png,
    gaussian_filter,
    io,
    np,
    plt,
    tower_atlas_mip,
    tower_mips,
    tower_template_registration,
):
    TOWER_SIDES = {1: ("W", "#ffb000"), 2: ("E", "#5aa9ff")}  # sides of the atlas


    def picture_tower_sides(volume, sides, names):
        """Draw a volume with its two sides in two colors (PNG).

        `sides` is 1 or 2 on a grid over the field of view of the volume, and 0 outside
        the tower. `names` gives the letter and the color of each side: the letter goes
        beside the foot of its side.
        """
        # 40 pixels for each mm, as (width, height).
        size = [round(40 * volume.sizes[dim] * volume.fusi.spacing[dim]) for dim in "ij"]

        def enlarge(values):
            """Values from 0 to 1 on a grid, at the size of the picture."""
            image = Image.fromarray(np.uint8(255 * values))
            return np.array(image.resize(size, Image.Resampling.BICUBIC)) / 255

        # Shades from 0 to 1, over the top 22 dB of the volume.
        power = volume.values[0] / volume.values.max()
        shade = enlarge(np.clip(1 + 10 * np.log10(power) / 22, 0, 1))
        colors = np.ones((*shade.shape, 3))
        letters = []
        for side, (letter, color) in names.items():
            # A little blur takes the voxel steps out of the edge of the side.
            mask = enlarge(gaussian_filter(np.float32(sides == side), 0.8))
            rgb = np.array([int(color[c : c + 2], 16) for c in (1, 3, 5)]) / 255
            colors += mask[..., None] * (rgb - 1)
            # The first side is on the left of the picture.
            rows, columns = np.nonzero(mask > 0.5)
            foot = columns[rows > rows.max() - 40]
            beside = foot.min() - 26 if side == 1 else foot.max() + 26
            letters.append(((beside, rows.max() - 14), letter, color))
        picture = Image.fromarray(np.uint8(255 * shade[..., None] * colors))
        draw = ImageDraw.Draw(picture)
        for position, letter, color in letters:
            draw.text(
                position, letter, fill=color, anchor="mm", font=ImageFont.load_default(40)
            )
        png = io.BytesIO()
        picture.save(png, format="PNG")
        return png.getvalue()


    def draw_tower_views():
        """Draw the tower from above, with the two views of it (PNG).

        The template shows the south face of the tower. Our volume shows its north face.
        """
        fig, ax = plt.subplots(figsize=(4, 6), layout="constrained")
        for side, (letter, color) in TOWER_SIDES.items():
            sign = (-1, 1)[side - 1]
            ax.fill_between([0, sign], -1, 1, color=color, alpha=0.85, lw=0)
            ax.text(1.55 * sign, 0, letter, color=color, ha="center", va="center", size=34)
        # The four legs go to the top of the tower.
        for corner in (-1, 1):
            ax.plot([-1, 1], [-corner, corner], color=PANEL, lw=2)
        ax.add_patch(Rectangle((-0.3, -0.3), 0.6, 0.6, color=PANEL))
        for sign, name in ((-1, "Template"), (1, "Our volume")):
            ax.annotate(
                "",
                xy=(0, 1.25 * sign),
                xytext=(0, 2.35 * sign),
                arrowprops={"arrowstyle": "-|>", "color": "white", "lw": 2.5},
            )
            ax.text(0, 2.8 * sign, name, ha="center", va="center", size=24)
        ax.set(xlim=(-2.1, 2.1), ylim=(-3.15, 3.15), aspect="equal")
        ax.axis("off")
        return fig_png(fig)


    _template, _volume = tower_atlas_mip.atlas.template, tower_mips["other"]
    _sides = tower_atlas_mip.atlas.hemispheres.where(
        tower_atlas_mip.atlas.annotation > 0, 0
    )
    # ConfUSIus brings the atlas into the world space of the volume. The grid is finer
    # than the volume in the plane of the picture, for a clean line between the two sides.
    _finer = {"k": 1, "j": 2, "i": 4}
    _spacing = {dim: _volume.fusi.spacing[dim] / _finer[dim] for dim in "kji"}
    _in_world = tower_atlas_mip.atlas.resample(
        np.linalg.inv(tower_template_registration[3]),
        output_sizes={dim: _volume.sizes[dim] * _finer[dim] for dim in "kji"},
        output_spacing=_spacing,
        output_origin={
            axis: _volume.fusi.origin[axis]
            + (_spacing[dim] - _volume.fusi.spacing[dim]) / 2
            for axis, dim in zip("zyx", "kji")
        },
        output_direction=_volume.fusi.direction,
    )
    tower_side_pngs = {
        "template": picture_tower_sides(_template, _sides.values[0], TOWER_SIDES),
        "views": draw_tower_views(),
        # Our volume shows the other face of the tower: its west is where the atlas has
        # its east. The registration cannot see it, because the tower is symmetric.
        "volume": picture_tower_sides(
            _volume,
            _in_world.hemispheres.where(_in_world.annotation > 0, 0).values[0],
            {1: TOWER_SIDES[2], 2: TOWER_SIDES[1]},
        ),
    }
    return (tower_side_pngs,)


@app.cell(hide_code=True)
def _(ACCENT, PANEL, Rectangle, fig_png, mo, plt):
    @mo.persistent_cache
    def render_space_definitions():
        """Draw a line of ten voxels in three kinds of space (one PNG for each space).

        Each picture writes, in each voxel, its position in one space.
        """
        first = 2.0  # center of the first voxel, in voxel widths from the left
        step = 0.2  # size of a voxel, mm

        def render(numbers, origin=None):
            fig, ax = plt.subplots(figsize=(10.36, 1.75))
            fig.subplots_adjust(0, 0, 1, 1)
            for voxel, number in enumerate(numbers):
                left = first + voxel - 0.5
                ax.add_patch(Rectangle((left + 0.06, 0.06), 0.88, 0.88, fc=PANEL, ec="w"))
                ax.text(first + voxel, 0.5, number, ha="center", va="center", size=17)
            if origin:
                # The origin of the space, with the direction of its axis.
                x, y, direction = origin
                ax.plot(x, y, "+", color=ACCENT, ms=20, mew=3)
                ax.annotate(
                    "",
                    xy=(x + 0.75 * direction, y),
                    xytext=(x + 0.25 * direction, y),
                    arrowprops={"arrowstyle": "-|>", "color": ACCENT, "lw": 2},
                )
                ax.text(13.6, 0.5, "mm", ha="left", va="center", size=19)
            # The same room above and below the voxels: an origin can go under them.
            ax.set(xlim=(0, 14.8), ylim=(-0.75, 1.75), aspect="equal")
            ax.axis("off")
            return fig_png(fig)

        def millimetres(origin, direction):
            """The position of the center of each voxel, from an origin along an axis."""
            return [
                f"{direction * (first + voxel - origin) * step:.1f}" for voxel in range(10)
            ]

        middle = first + 4.5
        return {
            "Voxel space": render([str(voxel) for voxel in range(10)]),
            # The origin of the world is beside the first voxel.
            "World space": render(millimetres(0.5, 1), origin=(0.5, 0.5, 1)),
            # A reference space has its own origin, here in the middle of the voxels, and
            # its axis can point the other way.
            "Reference space": render(millimetres(middle, -1), origin=(middle, -0.4, -1)),
        }


    space_definition_pngs = render_space_definitions()
    return (space_definition_pngs,)


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
        <div style="min-width:0; padding:.5rem; border:1px solid #2b3548; border-radius:12px; background:#111720; text-align:center">
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

    <div style="display:flex; gap:.6rem; margin:1.5rem 0">
      <div style="flex:1; min-width:0; padding:1rem; border:2px solid #e94b5f; border-radius:12px; background:#111720"><strong>① Acquire</strong></div>
      <div style="align-self:center; color:#f5c2ca">→</div>
      <div style="flex:1; min-width:0; padding:1rem; border:2px solid #e94b5f; border-radius:12px; background:#111720"><strong>② Reconstruct</strong></div>
      <div style="align-self:center; color:#f5c2ca">→</div>
      <div style="flex:1; min-width:0; padding:1rem; border:2px solid #e94b5f; border-radius:12px; background:#111720"><strong>③ Monitor</strong></div>
      <div style="align-self:center; color:#f5c2ca">→</div>
      <div style="flex:1; min-width:0; padding:1rem; border:2px solid #e94b5f; border-radius:12px; background:#111720"><strong>④ Store</strong></div>
    </div>

    | Scanner | Software | fUSI workflow |
    | :-- | :-- | :--: |
    | Iconeus One | Iconeus, bundled | <span style="color:#3ad9a4">✓</span> |
    | Verasonics | AUTC (proprietary) / EffUSIve (open-source, alpha) | <span style="color:#3ad9a4">✓</span> |
    | VisualSonics Vevo F2xc | VisualSonics | <span style="color:#ffd43b">?</span> |
    | us4us us4R | Your own / third-party software | <span style="color:#ffd43b">?</span> |
    | OpenSonics OpenExplorer | Your own / third-party software | <span style="color:#ffd43b">?</span> |
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    mo.md(f"""
    # Rodent neuronavigation

    <div style="display:flex; align-items:center; gap:.8rem; margin:.5rem 0">
      <div style="flex:1; min-width:0">{mo.image(Path("assets/navigation-rodent-bps.webp"), width="100%", style={"margin":"0", "display":"block"}).text}</div>
      <div style="flex:1; min-width:0; display:grid; gap:.6rem">
        <div style="padding:.7rem; border:1px solid #2b3548; border-radius:12px; background:#111720"><strong>BPS (mice)</strong><br>3D angiogram → vascular template / atlas → target 2D plane. Other setups may need manual positioning; check species/template availability.</div>
        <div style="padding:.7rem; border:1px solid #2b3548; border-radius:12px; background:#111720"><strong>QC every fit</strong><br>Check vessels, correct or reject bad matches, log probe pose. Expert landmarks can beat unchecked automation for comparable 2D slices.</div>
      </div>
    </div>

    <small>Nouhoum et al. (2021), [Fig. 1](https://doi.org/10.1038/s41598-021-94764-7) · CC BY 4.0.</small>
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    mo.md(f"""
    # Data acquisition · Human neuronavigation

    <div style="display:flex; align-items:center; gap:1.2rem; margin:1rem 0">
      <div style="flex:1; min-width:0">{mo.image(Path("assets/navigation-human-tracking.webp"), width="100%", style={"margin":"0", "display":"block"}).text}</div>
      <div style="flex:1; min-width:0; display:grid; gap:.5rem">
        <div style="padding:.6rem; border:1px solid #2b3548; border-radius:12px; background:#111720">① Track probe · optical / EM</div>
        <div style="padding:.6rem; border:1px solid #2b3548; border-radius:12px; background:#111720">② Calibrate markers ↔ 2D/3D US plane</div>
        <div style="padding:.6rem; border:1px solid #2b3548; border-radius:12px; background:#111720">③ Register image to MRI / atlas</div>
      </div>
    </div>

    **In fUSI:** Brainlab (awake surgery, 2023) · custom tracked helmet (mobile, 2025). **Open route:** SlicerIGT + PLUS / OpenIGTLink.

    <small>Soloukey et al. (2025), [Fig. 1B](https://pmc.ncbi.nlm.nih.gov/articles/PMC12175903/) · CC BY 4.0.</small>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Data acquisition · Before recording

    <div style="display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem">
      <div style="padding:1rem; border:1px solid #2b3548; border-radius:12px; background:#111720"><strong>Biology &amp; access</strong><br>Species &amp; age · preparation / acoustic window · awake vs anaesthetised. Match conditions.</div>
      <div style="padding:1rem; border:1px solid #2b3548; border-radius:12px; background:#111720"><strong>Where &amp; how often?</strong><br>Target regions · 2D plane(s) vs 3D · probe pose · temporal sampling &amp; duration.</div>
      <div style="padding:1rem; border:1px solid #2b3548; border-radius:12px; background:#111720"><strong>Experimental plan</strong><br>Baseline &amp; controls · task/trial timing · randomise or counterbalance · enough independent animals.</div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Data acquisition · During recording

    <div style="display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.8rem">
      <div style="padding:1rem; border:1px solid #2b3548; border-radius:12px; background:#111720"><strong>Sync the clocks</strong><br>Stimulus &amp; behaviour timestamps ↔ ultrasound time. Test triggers &amp; clock alignment before the study.</div>
      <div style="padding:1rem; border:1px solid #2b3548; border-radius:12px; background:#111720"><strong>Keep the right data</strong><br>Power Doppler / IQ / RF (if supported) · storage capacity · reference angiogram · acquisition parameters &amp; metadata.</div>
    </div>
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
    # A live page: this slide needs the network.
    mo.vstack(
        [
            mo.md("# ConfUSIus dataset collection"),
            mo.Html(
                '<iframe src="https://confusius-datasets.s3.amazonaws.com/index.html" '
                'title="ConfUSIus dataset collection" '
                # The height follows the width, so it follows the slide and not the
                # window: the page never makes the slide scroll.
                'style="width: 100%; aspect-ratio: 2.7; border: 0; border-radius: 8px; '
                'background: white"></iframe>'
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo, space_definition_pngs):
    _meanings = {
        "Voxel space": "Index of the voxel",
        "World space": "For a given acquisition<br>e.g. probe space",
        "Reference space": "To be shared across acquisitions<br>e.g. templates, atlases",
    }


    def _line(name, meaning):
        """One space: its name and meaning, then ten voxels with their positions in it."""
        meaning = f"<span style='font-size: 0.75em; opacity: 0.6'>{meaning}</span>"
        picture = mo.image(space_definition_pngs[name], height="18cqh")
        text = mo.md(f"- **{name}**<br>{meaning}")
        return f"<div>{text.text}</div><div>{picture.text}</div>"


    _lines = "".join(_line(name, meaning) for name, meaning in _meanings.items())

    mo.vstack(
        [
            mo.md("# Registration · Definitions"),
            mo.Html(
                "<div style='display: grid; grid-template-columns: auto 1fr; "
                f"align-items: center; gap: 0 2rem'>{_lines}</div>"
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo, swap_stops, tower_space_pngs):
    def _stop(step):
        """The picture and the text for one stop, from voxel space to world space."""
        trail = " → ".join(
            f"**{name}**" if index == step else f"<span style='opacity: 0.45'>{name}</span>"
            for index, name in enumerate(["Voxel space", "World space"])
        )
        equation = [
            r"$$(i, j, k) = I(i, j, k)$$",
            r"$$(x_0, y_0, z_0) = (S \circ I)(i, j, k)$$",
        ][step]
        caption = ["Positions are voxel indices.", "Positions are in millimetres."][step]
        text = [
            mo.md(f"<span style='white-space: nowrap; font-size: 0.8em'>{trail}</span>"),
            mo.md(equation),
            mo.md(caption),
        ]
        return mo.hstack(
            [mo.image(tower_space_pngs[step], height="60cqh"), mo.vstack(text, gap=1.5)],
            justify="center",
            gap=4,
            align="center",
        )


    mo.vstack(
        [mo.md("# Registration · Voxel to world space"), swap_stops([_stop(0), _stop(1)])],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(FIXED, MOVING, fragment, mo, tower_registrations):
    _before = tower_registrations["translated"][0]


    def _colored(text, color):
        return f"<span style='color: {color}'>{text}</span>"


    def _world_space(color):
        """The world space of one volume, with its S in the color of the volume."""
        return mo.md(rf"$$(x_0, y_0, z_0) = \textcolor{{{color}}}{{S}}(i, j, k)$$")


    # The two volumes have the same voxel grid and the same S. Key press 1 asks the
    # question. Key press 2 shows the transform.
    mo.hstack(
        [
            mo.image(_before, height="66cqh"),
            mo.vstack(
                [
                    mo.md(_colored("Reference volume", FIXED)),
                    _world_space(FIXED),
                    mo.md(_colored("Second volume", MOVING)),
                    _world_space(MOVING),
                    fragment(
                        mo.md(
                            rf"How to modify $\textcolor{{{MOVING}}}{{S}}$ so that the "
                            f"{_colored('second', MOVING)} and {_colored('first', FIXED)} "
                            "volumes are aligned in the world space?"
                        ),
                        0,
                    ),
                    fragment(
                        mo.md(
                            r"$$(x_1, y_1, z_1) = "
                            rf"(T \circ \textcolor{{{MOVING}}}{{S}})(i, j, k)$$"
                        ),
                        1,
                    ),
                ],
                gap=1,
            ),
        ],
        justify="center",
        gap=4,
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo, tower_registrations, tower_warp_png):
    def _card(tower, name, *lines, warp=None):
        """One kind of transform: its film, beside its name and a few short lines.

        `warp` is one more picture, for the right of the film.
        """
        pictures = [tower_registrations[tower][1]] + [warp] * (warp is not None)
        text = mo.vstack([mo.md(f"**{name}**"), *map(mo.md, lines)], gap=0.5)
        return mo.callout(
            mo.hstack(
                [*(mo.image(picture, height="26cqh") for picture in pictures), text],
                justify="start",
                gap=1.5,
                align="center",
            )
        )


    _cards = [
        _card("translated", "Translation transform", "$N$ parameters"),
        _card(
            "rotated",
            "Rigid transform",
            "Translation + rotation",
            "$N + N(N-1)/2$ parameters",
        ),
        _card(
            "sheared",
            "Affine transform",
            "Rigid + zooms + shear",
            "$N + N^2$ parameters",
        ),
        _card(
            "bent",
            "Deformation fields",
            "$N$ per control point (B-spline),<br>"
            "up to $N$ per voxel (dense field, e.g. SyN)",
            # The arrows show the motion that the B-spline found for the moving tower.
            warp=tower_warp_png,
        ),
    ]

    mo.vstack(
        [
            mo.md("# Registration · Transforms"),
            mo.md("$N$ = spatial dimensions").style({"opacity": "0.6"}),
            mo.hstack(_cards[:2], widths="equal", align="stretch", gap=1),
            mo.hstack(_cards[2:], widths="equal", align="stretch", gap=1),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(
    TOWER_REGIONS,
    captioned,
    fragment,
    mo,
    row,
    tower_atlas_png,
    tower_template_registration,
):
    _film, _in_template, _in_world, _ = tower_template_registration
    # The names of the regions, from the top of the tower down, in their colors.
    _names = "<br>".join(
        f"<span style='color: {color}'>{name}</span>"
        for name, color in reversed(TOWER_REGIONS.values())
    )


    def _result(picture, caption):
        """One result of the registration: a small picture, beside its caption."""
        caption = f"<span style='font-size: 0.8em; white-space: nowrap'>{caption}</span>"
        return row([mo.image(picture, height="27cqh"), mo.md(caption)], "start", gap=1)


    # The slide starts with the template and its atlas. Key press 1 shows the registration
    # of a volume to the template. Key press 2 shows the two results.
    mo.vstack(
        [
            mo.md("# Registration · Templates and atlases"),
            row(
                [
                    row([mo.image(tower_atlas_png, height="60cqh"), mo.md(_names)], gap=1),
                    fragment(captioned(_film, "World space → Template space", "54cqh"), 0),
                    fragment(
                        mo.vstack(
                            [
                                _result(_in_template, "World space → Template space"),
                                _result(_in_world, "Atlas → World space"),
                            ],
                            gap=1,
                        ),
                        1,
                    ),
                ],
                gap=2.5,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(Path, captioned, fragment, mo, row, swap_stops, tower_side_pngs):
    # The template shows one face of the tower, and our volume shows the other face. Key
    # press 1 shows the two views, and key press 2 shows our volume. Key press 3 shows the
    # text. Key press 4 swaps the towers for the same question in a fUSI experiment.
    _towers = row(
        [
            captioned(tower_side_pngs["template"], "Template"),
            fragment(captioned(tower_side_pngs["views"], "&nbsp;"), 0),
            fragment(captioned(tower_side_pngs["volume"], "Our volume"), 1),
        ],
        gap=4,
    )
    _experiment = row(
        [mo.image(Path("assets/orientation-qform-sform.png"), height="55cqh")]
    )

    mo.vstack(
        [
            mo.md("# Registration · What about orientation?"),
            swap_stops([_towers, _experiment], start=3),
            fragment(
                mo.md(
                    "The orientation of the data and the definition of the spaces matter."
                ),
                2,
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
                        "Each volume time point is registered to a reference volume.",
                    ),
                    _part(
                        1,
                        "Across sessions and animals",
                        sessions_pngs["pair"],
                        "The same coordinates must show the same region.",
                    ),
                    _part(
                        2,
                        "Registration to templates",
                        atlas_png,
                        "Registration through pre-aligned power Doppler templates",
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
    mo.md("""
    # Denoising · Overview

    <div style="display:flex; gap:.6rem; margin:1.2rem 0">
      <div style="flex:1; min-width:0; padding:.85rem; border:2px solid #e94b5f; border-radius:12px; background:#111720">
        <strong>① Clutter filter</strong><br><span style="color:#9fb7bc; font-size:.82em">IQ → Power Doppler</span>
      </div>
      <div style="align-self:center; color:#f5c2ca">→</div>
      <div style="flex:1; min-width:0; padding:.85rem; border:2px solid #e94b5f; border-radius:12px; background:#111720">
        <strong>② Scrubbing</strong><br><span style="color:#9fb7bc; font-size:.82em">Censor bursts</span>
      </div>
      <div style="align-self:center; color:#f5c2ca">→</div>
      <div style="flex:1; min-width:0; padding:.85rem; border:2px solid #e94b5f; border-radius:12px; background:#111720">
        <strong>③ Frequency filter</strong><br><span style="color:#9fb7bc; font-size:.82em">Drift &amp; physiology</span>
      </div>
      <div style="align-self:center; color:#f5c2ca">→</div>
      <div style="flex:1; min-width:0; padding:.85rem; border:2px solid #e94b5f; border-radius:12px; background:#111720">
        <strong>④ Regression</strong><br><span style="color:#9fb7bc; font-size:.82em">Residual confounds</span>
      </div>
    </div>

    <div style="display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem">
      <div style="padding:.85rem; border:1px solid #2b3548; border-radius:12px; background:#111720"><strong>Noise sources</strong><br>Tissue motion · acoustic flashes · heart / breathing · slow drift</div>
      <div style="padding:.85rem; border:1px solid #2b3548; border-radius:12px; background:#111720"><strong>Order matters</strong><br>Handle spikes first · filter confounds like signal <em>(Lindquist 2019)</em></div>
      <div style="padding:.85rem; border:1px solid #2b3548; border-radius:12px; background:#111720"><strong>Context &amp; QC</strong><br>Task vs. rsFC · awake vs. anaesthetised · QC before &amp; after</div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Denoising · Clutter filtering

    Raw IQ → Power Doppler: separate **blood flow** from **tissue echoes**.

    | Approach | Mechanism | Trade-off |
    | :-- | :-- | :-- |
    | **Butterworth (high-pass)** | Doppler velocity | Useless alone in awake fUSI — combine with SVD |
    | **Static SVD** | Spatiotemporal coherence, fixed rank | Simple; one cutoff for all frames |
    | **Adaptive SVD** | Per-packet cutoff (e.g. constant-energy) + mask | Best motion robustness — no perfect heuristic yet |

    <div style="margin-top:.9rem; color:#9fb7bc; font-size:.88em">
      <strong style="color:#f5c2ca">Tools:</strong> IcoStudio / IcoLab · AUTC · EffUSIve · PyfUS &nbsp;<em>(requires raw IQ or scanner support)</em>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(fig_png, mo, np, plt):
    _rng = np.random.default_rng(7)
    _fs = 2.5
    _t = np.arange(0, 120, 1 / _fs)
    _drift = 0.9 * np.sin(2 * np.pi * 0.004 * _t)
    _neuro = sum(
        _rng.uniform(0.3, 0.7) * np.sin(2 * np.pi * _f * _t + _rng.uniform(0, 2 * np.pi))
        for _f in (0.025, 0.045, 0.07)
    )
    _physio = 1.4 * np.sin(2 * np.pi * 0.65 * _t) + 1.1 * np.sin(2 * np.pi * 1.1 * _t)
    _sig = _drift + _neuro + _physio + _rng.normal(0, 0.25, len(_t))
    _freqs = np.fft.rfftfreq(len(_sig), d=1 / _fs)
    _mag = np.abs(np.fft.rfft(_sig))

    _fig, _ax = plt.subplots(figsize=(5.6, 3.4))
    _ax.set_facecolor("none")
    _ax.plot(_freqs, _mag, color="#3ad9a4", lw=1.8)
    _ax.axvspan(0, 0.01, color="#e94b5f", alpha=0.45, label="Drift (<0.01 Hz)")
    _ax.axvspan(0.1, _fs / 2, color="#f0b429", alpha=0.16, label="High-freq (>0.1 Hz)")
    _ax.axvline(0.01, color="#e94b5f", ls="--", lw=1.2)
    _ax.axvline(0.1, color="#f0b429", ls="--", lw=1.2)
    _ax.set_xlim(0, _fs / 2)
    _ax.set_ylim(0, _mag.max() * 1.12)
    _ax.set_xlabel("Frequency (Hz)", color="#c9d1d9")
    _ax.set_ylabel("Magnitude", color="#c9d1d9")
    _ax.tick_params(colors="#9fb7bc")
    for _s in ("bottom", "left"):
        _ax.spines[_s].set_color("#4b5563")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend(frameon=False, fontsize=11, loc="upper right", labelcolor="#f8f9fb")
    _fig.tight_layout()
    _spec_png = fig_png(_fig)

    mo.md(f"""
    # Denoising · Frequency filtering

    <div style="display:flex; align-items:center; gap:1.3rem; margin:.8rem 0">
      <div style="flex:1.15; min-width:0">{mo.image(_spec_png, width="100%", style={"margin":"0", "display":"block"}).text}</div>
      <div style="flex:1; min-width:0; display:flex; flex-direction:column; gap:.65rem">
        <div style="padding:.75rem .9rem; border:1px solid #2b3548; border-radius:12px; background:#111720">
          <strong>High-pass (~0.01 Hz)</strong><br>Cuts slow drifts; keeps broadband signal.
        </div>
        <div style="padding:.75rem .9rem; border:1px solid #2b3548; border-radius:12px; background:#111720">
          <strong>Band-pass (0.01–0.1 Hz)</strong><br>Cuts physiology; low-pass can smear motion spikes.
        </div>
        <div style="padding:.75rem .9rem; border:1px solid #2b3548; border-radius:12px; background:#111720">
          <strong>Watch out</strong><br>Zero-phase · keep task frequencies · mind aliasing.
        </div>
      </div>
    </div>

    <div style="color:#9fb7bc; font-size:.86em">
      <strong style="color:#f5c2ca">Tools:</strong> ConfUSIus · Nilearn · SciPy · IcoStudio / IcoLab
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(fig_png, mo, np, plt):
    _rng = np.random.default_rng(12)
    _t = np.linspace(0, 120, 300)
    _clean = 3.2 * np.sin(2 * np.pi * 0.035 * _t) + 1.8 * np.cos(2 * np.pi * 0.06 * _t) + _rng.normal(0, 0.9, len(_t))
    _confound = (
        18.0 * np.exp(-0.5 * ((_t - 34) / 2.2) ** 2)
        + 24.0 * np.exp(-0.5 * ((_t - 82) / 2.8) ** 2)
        + 4.5 * np.sin(2 * np.pi * 0.008 * _t)
    )
    _raw = _clean + _confound

    _fig, _ax = plt.subplots(figsize=(5.6, 3.4))
    _ax.set_facecolor("none")
    _ax.plot(_t, _raw, color="#e94b5f", lw=1.6, alpha=0.9, label="Before regression")
    _ax.plot(_t, _clean, color="#3ad9a4", lw=1.8, label="After regression")
    _ax.fill_between(_t, _clean, _raw, color="#e94b5f", alpha=0.18, label="Removed confounds")
    _ax.set_xlim(0, 120)
    _ax.set_xlabel("Time (s)", color="#c9d1d9")
    _ax.set_ylabel("Signal (a.u.)", color="#c9d1d9")
    _ax.tick_params(colors="#9fb7bc")
    for _s in ("bottom", "left"):
        _ax.spines[_s].set_color("#4b5563")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend(frameon=False, fontsize=11, loc="upper left", labelcolor="#f8f9fb")
    _fig.tight_layout()
    _reg_png = fig_png(_fig)

    mo.md(f"""
    # Denoising · Confound regression

    <div style="display:flex; align-items:center; gap:1.3rem; margin:.8rem 0">
      <div style="flex:1.15; min-width:0">{mo.image(_reg_png, width="100%", style={"margin":"0", "display":"block"}).text}</div>
      <div style="flex:1; min-width:0; display:flex; flex-direction:column; gap:.65rem">
        <div style="padding:.75rem .9rem; border:1px solid #2b3548; border-radius:12px; background:#111720">
          <strong>Global Signal (GSR)</strong><br>Brain mean; shifts correlations, weak in awake fUSI.
        </div>
        <div style="padding:.75rem .9rem; border:1px solid #e94b5f; border-radius:12px; background:#111720">
          <strong>CompCor (aCompCor / tCompCor)</strong><br>Noise-ROI PCs; <strong>aCompCor</strong> leads in awake benchmarks.
        </div>
        <div style="padding:.75rem .9rem; border:1px solid #2b3548; border-radius:12px; background:#111720">
          <strong>Motion &amp; physiology</strong><br>Registration params · tissue velocity · cardiac / resp.
        </div>
      </div>
    </div>

    <div style="color:#9fb7bc; font-size:.86em">
      <strong style="color:#f5c2ca">Tools:</strong> ConfUSIus · Nilearn · IcoStudio / IcoLab &nbsp;<em>(task fUSI: include confounds in the GLM)</em>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(fig_png, mo, np, plt):
    _rng = np.random.default_rng(42)
    _t = np.linspace(0, 120, 300)
    _sig = 3.0 * np.sin(2 * np.pi * 0.03 * _t) + _rng.normal(0, 2.0, len(_t))
    for _center, _amp in ((32, 34), (78, 42), (83, 28)):
        _sig += _amp * np.exp(-0.5 * ((_t - _center) / 1.4) ** 2)
    _thresh = 15.0
    _bad = _sig > _thresh

    _fig, _ax = plt.subplots(figsize=(5.6, 3.4))
    _ax.set_facecolor("none")
    _ax.plot(_t, _sig, color="#3ad9a4", lw=1.6, label="Signal / QC metric")
    _ax.axhline(_thresh, color="#f5c2ca", ls="--", lw=1.3, label="Threshold")
    _ax.fill_between(_t, -10, 55, where=_bad, color="#e94b5f", alpha=0.32, label="Censored frames")
    _ax.set_xlim(0, 120)
    _ax.set_ylim(-10, 55)
    _ax.set_xlabel("Time (s)", color="#c9d1d9")
    _ax.set_ylabel("Relative CBV (%)", color="#c9d1d9")
    _ax.tick_params(colors="#9fb7bc")
    for _s in ("bottom", "left"):
        _ax.spines[_s].set_color("#4b5563")
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend(frameon=False, fontsize=11, loc="upper left", labelcolor="#f8f9fb")
    _fig.tight_layout()
    _scrub_png = fig_png(_fig)

    mo.md(f"""
    # Denoising · Scrubbing

    <div style="display:flex; align-items:center; gap:1.3rem; margin:.8rem 0">
      <div style="flex:1.15; min-width:0">{mo.image(_scrub_png, width="100%", style={"margin":"0", "display":"block"}).text}</div>
      <div style="flex:1; min-width:0; display:flex; flex-direction:column; gap:.65rem">
        <div style="padding:.75rem .9rem; border:1px solid #2b3548; border-radius:12px; background:#111720">
          <strong>Last-resort censoring</strong><br>Drop frames with residual motion spikes.
        </div>
        <div style="padding:.75rem .9rem; border:1px solid #2b3548; border-radius:12px; background:#111720">
          <strong>Detection metrics</strong><br>rDVARS · rGS · FD · tissue velocity (GV).
        </div>
        <div style="padding:.75rem .9rem; border:1px solid #e94b5f; border-radius:12px; background:#111720">
          <strong>Often optional</strong><br>Adaptive SVD + aCompCor lets you keep more data.
        </div>
      </div>
    </div>

    <div style="color:#9fb7bc; font-size:.86em">
      <strong style="color:#f5c2ca">Tools:</strong> ConfUSIus · Nilearn · IcoStudio / IcoLab
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Denoising · Software

    <div style="display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.9rem; margin:1.3rem 0">
      <div style="padding:1rem; border:1px solid #2b3548; border-radius:12px; background:#111720">
        <strong>IcoStudio / IcoLab</strong><br><br>
        Built-in, curated preprocessing.<br>
        <span style="color:#9fb7bc; font-size:.88em">Proprietary; pipeline not documented online.</span>
      </div>
      <div style="padding:1rem; border:1px solid #2b3548; border-radius:12px; background:#111720">
        <strong>Nilearn</strong><br><br>
        fMRI standard adapted to fUSI in the literature.<br>
        <span style="color:#9fb7bc; font-size:.88em">Lindquist et al. order; some defaults fit fMRI better than fUSI.</span>
      </div>
      <div style="padding:1rem; border:2px solid #e94b5f; border-radius:12px; background:#111720">
        <strong>ConfUSIus</strong><br><br>
        Adapts Nilearn’s pipeline for fUSI (Lindquist et al.).<br>
        <span style="color:#f5c2ca; font-size:.88em">Fixes fUSI pitfalls (e.g. long cubic-spline interpolations).</span>
      </div>
    </div>
    """)
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


if __name__ == "__main__":
    app.run()
