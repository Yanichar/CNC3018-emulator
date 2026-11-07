"""Render the work-field PNG without opening the GUI.

Example::

    python tools/render_workfield.py --gerber-dir C:\\work\\ecc83\\gerber

The script loads every ``*.drl`` file from the directory, takes the Gerber
layer that looks like the board outline (or ``--layer``) and writes the
20 px/mm image used by the future virtual camera.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grbl_emulator.pcb import (  # noqa: E402
    PcbError,
    discover_sources,
    guess_outline_layer,
    load_board,
)
from grbl_emulator.workfield import (  # noqa: E402
    FieldGeometry,
    center_offset,
    default_output_path,
    render_workfield,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gerber-dir", required=True, help="directory with Gerber and .drl files")
    parser.add_argument("--layer", help="outline Gerber file name (default: auto-detect)")
    parser.add_argument(
        "--output",
        default=str(default_output_path()),
        help=f"PNG path (default: {default_output_path()})",
    )
    args = parser.parse_args(argv)

    directory = Path(args.gerber_dir)
    try:
        drills, gerbers = discover_sources(directory)
    except OSError as exc:
        print(f"Cannot read {directory}: {exc}", file=sys.stderr)
        return 1
    if not gerbers:
        print(f"No Gerber files in {directory}", file=sys.stderr)
        return 1

    names = [path.name for path in gerbers]
    layer = args.layer or guess_outline_layer(names)
    if layer not in names:
        print(f"Layer {layer!r} is not in {directory}", file=sys.stderr)
        print("Available layers: " + ", ".join(names), file=sys.stderr)
        return 1
    gerber = next(path for path in gerbers if path.name == layer)

    try:
        board = load_board(gerber, drills)
    except PcbError as exc:
        print(f"Cannot load board: {exc}", file=sys.stderr)
        return 1

    geometry = FieldGeometry()
    output = render_workfield(
        Path(args.output), board, center_offset(board, geometry), geometry
    )
    print(
        f"Board {board.width:.2f} x {board.height:.2f} mm, {len(board.holes)} holes, "
        f"layer {board.source}"
    )
    print(
        f"Work field {geometry.width_mm:g} x {geometry.height_mm:g} mm -> "
        f"{output} ({geometry.image_width}x{geometry.image_height} px, "
        f"{geometry.px_per_mm:g} px/mm)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
