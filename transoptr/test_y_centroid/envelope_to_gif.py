"""
envelope_to_gif.py
==================
Reads a folder of fort.envelope.* files (one per optimization step),
renders each as a beam-profile PNG in the style of Beamline._generate_simulation,
attaches a text side-panel with step info and summary stats, and stitches
the frames into an animated GIF.

Memory strategy
---------------
* Files are processed one at a time — only 5 columns are loaded per file.
* Each frame is rendered to a temporary PNG on disk, then matplotlib is closed.
* The GIF is assembled by streaming frames from disk via a generator,
  so at most ~2 frames are resident in RAM at any time.

Usage
-----
    python envelope_to_gif.py --envelope_dir ./envelope \
                              --output tuning_results.gif \
                              --thin 10 \
                              --dpi 100 \
                              --duration 300 \
                              --wall_width 3.0
"""

import argparse
import glob
import os
import re
import tempfile
import shutil
import gc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no GUI, less memory
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# Columns we need (0-indexed positions in the header)
NEEDED_COLUMNS = ['s', 'x-envelope', 'y-envelope', 'x-centroid', 'y-centroid']


def parse_envelope_file(filepath):
    """
    Parse a fort.envelope file, extracting only the columns we need.

    Returns
    -------
    dict with keys: 's', 'x_ev', 'y_ev', 'x_cm', 'y_cm' — all numpy arrays
    """
    with open(filepath, 'r') as f:
        header_line = f.readline()
        _units_line = f.readline()  # skip units row

        # figure out which column indices we need
        headers = header_line.split()
        col_indices = []
        for name in NEEDED_COLUMNS:
            try:
                col_indices.append(headers.index(name))
            except ValueError:
                raise ValueError(
                    f"Column '{name}' not found in {filepath}. "
                    f"Available columns: {headers[:10]}..."
                )

        # read only the columns we need — line by line to keep memory flat
        rows = []
        for line in f:
            parts = line.split()
            if len(parts) < max(col_indices) + 1:
                continue
            rows.append([float(parts[i]) for i in col_indices])

    data = np.array(rows, dtype=np.float64)

    # Clean out spurious all-zero rows that appear mid-file.
    # These cause the plot lines to jump back to s=0.
    # Keep the first row even if it's genuinely at s=0.
    all_zero = np.all(data == 0.0, axis=1)
    # preserve row 0 (legitimate starting point)
    if len(all_zero) > 0:
        all_zero[0] = False
    data = data[~all_zero]

    return {
        's':    data[:, 0],
        'x_ev': data[:, 1],
        'y_ev': data[:, 2],
        'x_cm': data[:, 3],
        'y_cm': data[:, 4],
    }


# ---------------------------------------------------------------------------
# Plotting — mirrors Beamline._generate_simulation style
# ---------------------------------------------------------------------------

def render_beam_profile(data, step_number, fig_width=16, fig_height=5, dpi=100,
                        wall_width=3.0,
                        slit_data=None,
                        transmission_data=None,
                        extra_overlay_fn=None):
    """
    Render a single beam-profile frame and return it as a PIL Image.

    Parameters
    ----------
    data : dict
        Output of parse_envelope_file.
    step_number : int
        Optimization step index (for labelling).
    wall_width : float
        Y-axis half-range (cm). Matches Beamline.wall_width.
    slit_data : dict or None
        HOOK — if provided, should contain:
            'x_slits' : array same length as s
            'y_slits' : array same length as s
        These will be plotted as step-functions (upper x, lower y).
    transmission_data : dict or None
        HOOK — if provided, should contain:
            'propagated_transmissions' : array same length as s
        Will be plotted as a green filled step function.
    extra_overlay_fn : callable or None
        HOOK — signature: extra_overlay_fn(ax, data, step_number)
        Lets you draw anything else (steerer lines, FC markers, etc.)

    Returns
    -------
    PIL.Image.Image  (RGB)
    """
    s    = data['s']
    x_cm = data['x_cm']
    y_cm = data['y_cm']
    x_ev = data['x_ev']
    y_ev = data['y_ev']

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_subplot(111)

    # --- X plane (blue) ---
    ax.plot(s, x_cm, color='b', label='x plane')
    ax.plot(s, x_cm + x_ev, linestyle='dashed', color='b')
    ax.plot(s, x_cm - x_ev, linestyle='dashed', color='b')
    ax.fill_between(s, x_cm - x_ev, x_cm + x_ev, color='b', alpha=0.2)

    # --- Y plane (red) ---
    ax.plot(s, y_cm, color='r', label='y plane')
    ax.plot(s, y_cm + y_ev, linestyle='dashed', color='r')
    ax.plot(s, y_cm - y_ev, linestyle='dashed', color='r')
    ax.fill_between(s, y_cm - y_ev, y_cm + y_ev, color='r', alpha=0.2)

    # --- Centre line ---
    ax.axhline(0, linestyle='solid', color='gray', alpha=0.7)

    # --- HOOK: slit overlay ---
    if slit_data is not None:
        ax.step(s, slit_data['x_slits'], where='mid', color='b', alpha=0.5, label='slit radius x')
        ax.step(s, -slit_data['y_slits'], where='mid', color='r', alpha=0.5, label='slit radius y')

    # --- HOOK: transmission overlay ---
    if transmission_data is not None:
        ax.axhline(1, linestyle='dashed', color='black', linewidth=1)
        ax.step(s, transmission_data['propagated_transmissions'],
                color='green', where='post', label='transmissions')
        ax.fill_between(s, 0, transmission_data['propagated_transmissions'],
                        color='green', alpha=0.4)

    # --- HOOK: arbitrary extra overlays ---
    if extra_overlay_fn is not None:
        extra_overlay_fn(ax, data, step_number)

    ax.set_xlim(-10, s[-1] + 30)
    ax.set_ylim(-wall_width, wall_width)
    ax.set_xlabel("Distance Along Beam Axis (cm)")
    ax.set_ylabel("Distance in Radial Direction (cm)")
    ax.legend(loc='lower left')
    fig.tight_layout()

    # Rasterise to numpy → PIL (same technique as Beamline.render)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
    w, h = fig.canvas.get_width_height()
    buf = buf.reshape(h, w, 4)
    img = Image.fromarray(buf[:, :, :3])  # drop alpha → RGB

    plt.close(fig)
    del buf
    return img


# ---------------------------------------------------------------------------
# Summary statistics for the side panel
# ---------------------------------------------------------------------------

def compute_summary_stats(data):
    """
    Return a list of human-readable strings summarising one envelope snapshot.
    Extend this when transmission / slit data is available.
    """
    s    = data['s']
    x_cm = data['x_cm']
    y_cm = data['y_cm']
    x_ev = data['x_ev']
    y_ev = data['y_ev']

    stats = [
        f"s range: {s[0]:.1f} - {s[-1]:.1f} cm",
        f"max x-env: {np.max(x_ev):.4f} cm",
        f"max y-env: {np.max(y_ev):.4f} cm",
        f"final x-cen: {x_cm[-1]:.4f} cm",
        f"final y-cen: {y_cm[-1]:.4f} cm",
        f"final x-env: {x_ev[-1]:.4f} cm",
        f"final y-env: {y_ev[-1]:.4f} cm",
    ]
    return stats


def compute_transmission_stats(transmission_data):
    """
    HOOK — call this and append its output to the blurb when transmission
    data becomes available.
    """
    t = transmission_data['propagated_transmissions']
    return [
        f"final trans: {t[-1]:.4f}",
        f"min trans:   {np.min(t):.4f}",
    ]


# ---------------------------------------------------------------------------
# Text side-panel (adapted from plotting.py _save_gif)
# ---------------------------------------------------------------------------

def make_combined_frame(beam_img, blurb, fontsize=0.8, font_thickness=1):
    """
    Paste a text panel to the right of beam_img.
    beam_img : PIL.Image.Image (RGB)
    blurb    : list[str]
    Returns  : PIL.Image.Image (RGB)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_sizes = [cv2.getTextSize(line, font, fontsize, font_thickness)[0] for line in blurb]
    max_text_w = int(max(tw for tw, _ in text_sizes) * 1.3)
    panel_h = beam_img.size[1]  # PIL .size is (w, h)

    textbox = np.ones((panel_h, max_text_w, 3), dtype=np.uint8) * 255
    line_height = max(th for _, th in text_sizes) + 8
    y_cursor = line_height + 10
    x_margin = 10

    for line in blurb:
        cv2.putText(textbox, line, (x_margin, y_cursor), font,
                    fontsize, (0, 0, 0), font_thickness, cv2.LINE_AA)
        y_cursor += line_height

    textbox_img = Image.fromarray(textbox)
    combined = Image.new('RGB', (beam_img.size[0] + textbox_img.size[0], panel_h), (250, 250, 250))
    combined.paste(beam_img, (0, 0))
    combined.paste(textbox_img, (beam_img.size[0], 0))
    return combined


# ---------------------------------------------------------------------------
# File discovery & sorting
# ---------------------------------------------------------------------------

def discover_envelope_files(envelope_dir):
    """
    Find all fort.envelope.* files, return sorted by step number.
    Expects filenames like fort.envelope.00001 or fort.envelope.1
    """
    pattern = os.path.join(envelope_dir, 'fort.envelope.*')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No fort.envelope.* files found in {envelope_dir}")

    def sort_key(path):
        basename = os.path.basename(path)
        # extract the numeric suffix after the last '.'
        suffix = basename.rsplit('.', 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            return 0

    files.sort(key=sort_key)
    return files


def extract_step_number(filepath):
    """Pull the integer step number from fort.envelope.XXXXX"""
    suffix = os.path.basename(filepath).rsplit('.', 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_gif(envelope_dir, output_path, thin=1, dpi=100, duration=30,
                 wall_width=3.0, fig_width=16, fig_height=5,
                 slit_loader_fn=None, transmission_loader_fn=None,
                 extra_overlay_fn=None):
    """
    Main entry point.

    Parameters
    ----------
    envelope_dir : str
        Directory containing fort.envelope.* files.
    output_path : str
        Output .gif file path.
    thin : int
        Use every k-th file (1 = all files).
    dpi : int
        Resolution of each frame.
    duration : int
        Milliseconds per frame in the GIF.
    wall_width : float
        Y-axis half-range for the beam profile plot.
    slit_loader_fn : callable or None
        HOOK — signature: slit_loader_fn(step_number) -> dict or None
        Should return {'x_slits': array, 'y_slits': array} or None.
    transmission_loader_fn : callable or None
        HOOK — signature: transmission_loader_fn(step_number) -> dict or None
        Should return {'propagated_transmissions': array} or None.
    extra_overlay_fn : callable or None
        HOOK — signature: extra_overlay_fn(ax, data, step_number)
    """
    print(f"Discovering envelope files in: {envelope_dir}")
    all_files = discover_envelope_files(envelope_dir)
    total = len(all_files)
    print(f"Found {total} envelope files")

    # apply thinning
    selected_files = all_files[::thin]
    n_frames = len(selected_files)
    print(f"After thinning (k={thin}): {n_frames} frames to render")

    # temporary directory for per-frame PNGs
    tmp_dir = tempfile.mkdtemp(prefix='envelope_gif_')
    print(f"Temp frame directory: {tmp_dir}")

    try:
        # --- Pass 1: render each frame to a temp PNG ---
        frame_paths = []
        for idx, filepath in enumerate(selected_files):
            step = extract_step_number(filepath)

            # parse — only 5 columns loaded
            data = parse_envelope_file(filepath)

            # load optional hook data
            slit_data = slit_loader_fn(step) if slit_loader_fn else None
            transmission_data = transmission_loader_fn(step) if transmission_loader_fn else None

            # render beam profile
            beam_img = render_beam_profile(
                data, step,
                fig_width=fig_width, fig_height=fig_height, dpi=dpi,
                wall_width=wall_width,
                slit_data=slit_data,
                transmission_data=transmission_data,
                extra_overlay_fn=extra_overlay_fn,
            )

            # build blurb for side panel
            blurb = [f"STEP: {step}"]
            blurb += compute_summary_stats(data)
            if transmission_data is not None:
                blurb += compute_transmission_stats(transmission_data)

            # combine beam + text panel
            combined = make_combined_frame(beam_img, blurb)

            # save to disk immediately
            frame_path = os.path.join(tmp_dir, f"frame_{idx:06d}.png")
            combined.save(frame_path, format='PNG')
            frame_paths.append(frame_path)

            # free everything
            del data, beam_img, combined, slit_data, transmission_data
            if idx % 100 == 0:
                gc.collect()

            if (idx + 1) % max(1, n_frames // 20) == 0 or idx == n_frames - 1:
                print(f"  Rendered frame {idx + 1}/{n_frames}  (step {step})")

        # --- Pass 2: assemble GIF by streaming from disk ---
        print(f"Assembling GIF ({n_frames} frames) ...")

        def frame_generator():
            """Yield PIL Images one at a time from disk — keeps ~1 frame in RAM."""
            for fp in frame_paths[1:]:
                img = Image.open(fp)
                img.load()  # force read so file handle can close
                # quantize to palette to reduce GIF size
                yield img.quantize(colors=128, method=Image.Quantize.FASTOCTREE)

        first_frame = Image.open(frame_paths[0])
        first_frame.load()
        first_quantized = first_frame.quantize(colors=128, method=Image.Quantize.FASTOCTREE)

        first_quantized.save(
            output_path,
            format='GIF',
            save_all=True,
            append_images=frame_generator(),
            duration=duration,
            loop=0,
        )

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"GIF saved: {output_path}  ({file_size_mb:.1f} MB, {n_frames} frames)")

    finally:
        # clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("Temp files cleaned up.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert fort.envelope.* files to an animated beam-profile GIF."
    )
    parser.add_argument('--envelope_dir', type=str, default='./envelope',
                        help='Directory containing fort.envelope.* files')
    parser.add_argument('--output', type=str, default='beam_envelope.gif',
                        help='Output GIF file path')
    parser.add_argument('--thin', type=int, default=1,
                        help='Use every k-th file (1 = all files)')
    parser.add_argument('--dpi', type=int, default=100,
                        help='Resolution of each frame')
    parser.add_argument('--duration', type=int, default=30,
                        help='Milliseconds per frame in the GIF')
    parser.add_argument('--wall_width', type=float, default=3.0,
                        help='Y-axis half-range in cm')
    parser.add_argument('--fig_width', type=float, default=16,
                        help='Figure width in inches')
    parser.add_argument('--fig_height', type=float, default=5,
                        help='Figure height in inches')

    args = parser.parse_args()

    generate_gif(
        envelope_dir=args.envelope_dir,
        output_path=args.output,
        thin=args.thin,
        dpi=args.dpi,
        duration=args.duration,
        wall_width=args.wall_width,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
    )


if __name__ == '__main__':
    main()
