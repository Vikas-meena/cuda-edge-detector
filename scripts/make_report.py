#!/usr/bin/env python3
"""Turns the timing table into the artifacts shown in the README.

Reads results/timings.csv (written by bin/edge_detector) and produces:
  results/summary.md   per-resolution table of GPU and CPU timings
  results/speedup.png  bar chart of the same data
  results/montage.png  input images next to their edge maps
"""

import argparse
import collections
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

MONTAGE_TILE = 384  # Edge length of one montage tile, in pixels.
MONTAGE_ROWS = 4


def read_rows(csv_path):
    with open(csv_path, newline="") as handle:
        return list(csv.DictReader(handle))


def group_by_resolution(rows):
    """Returns {(width, height): [row, ...]} ordered by pixel count."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(int(row["width"]), int(row["height"]))].append(row)
    return dict(sorted(groups.items(), key=lambda item: item[0][0] * item[0][1]))


def mean(rows, column):
    values = [float(row[column]) for row in rows if row[column] != ""]
    return sum(values) / len(values) if values else None


def write_summary(groups, rows, path):
    has_cpu = any(row["cpu_ms"] != "" for row in rows)
    lines = [
        "# Run summary",
        "",
        f"Images processed in one execution: **{len(rows)}**",
        "",
        "| Resolution | Images | GPU upload ms | GPU kernels ms | "
        "GPU download ms | GPU total ms | CPU ms | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (width, height), group in groups.items():
        gpu = mean(group, "gpu_total_ms")
        cpu = mean(group, "cpu_ms")
        cpu_text = f"{cpu:.2f}" if cpu is not None else "n/a"
        speedup_text = f"{cpu / gpu:.1f}x" if cpu is not None and gpu else "n/a"
        lines.append(
            f"| {width}x{height} | {len(group)} | "
            f"{mean(group, 'gpu_upload_ms'):.3f} | "
            f"{mean(group, 'gpu_compute_ms'):.3f} | "
            f"{mean(group, 'gpu_download_ms'):.3f} | {gpu:.3f} | "
            f"{cpu_text} | {speedup_text} |"
        )
    lines.append("")
    lines.append("Timings are per-image means within each resolution.")
    if has_cpu:
        mismatches = sum(int(row["mismatched_pixels"]) for row in rows)
        pixels = sum(int(row["width"]) * int(row["height"]) for row in rows)
        lines.append(
            f"GPU and CPU edge maps differ in **{mismatches}** of {pixels} pixels."
        )
    path.write_text("\n".join(lines) + "\n")


def write_chart(groups, path):
    labels = [f"{w}x{h}" for (w, h) in groups]
    gpu = [mean(group, "gpu_total_ms") for group in groups.values()]
    cpu = [mean(group, "cpu_ms") for group in groups.values()]
    positions = range(len(labels))
    width = 0.38

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([p - width / 2 for p in positions], gpu, width, label="GPU (CUDA)")
    if all(value is not None for value in cpu):
        axis.bar(
            [p + width / 2 for p in positions], cpu, width, label="CPU (1 thread)"
        )
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels)
    axis.set_yscale("log")
    axis.set_xlabel("Image resolution")
    axis.set_ylabel("Mean time per image (ms, log scale)")
    axis.set_title("Canny edge detection: GPU vs CPU")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def write_montage(rows, input_dir, output_dir, path):
    """Pairs each of the largest few images with its edge map."""
    largest = sorted(
        rows, key=lambda row: int(row["width"]) * int(row["height"]), reverse=True
    )[:MONTAGE_ROWS]
    montage = Image.new("RGB", (2 * MONTAGE_TILE, len(largest) * MONTAGE_TILE))
    for index, row in enumerate(largest):
        stem = pathlib.Path(row["image"]).stem
        pair = (input_dir / row["image"], output_dir / f"{stem}_edges.png")
        for column, image_path in enumerate(pair):
            with Image.open(image_path) as image:
                tile = image.convert("RGB").resize((MONTAGE_TILE, MONTAGE_TILE))
            montage.paste(tile, (column * MONTAGE_TILE, index * MONTAGE_TILE))
    montage.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default="results/timings.csv")
    parser.add_argument("--input_dir", default="data/input")
    parser.add_argument("--output_dir", default="data/output")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.csv)
    groups = group_by_resolution(rows)

    write_summary(groups, rows, results_dir / "summary.md")
    write_chart(groups, results_dir / "speedup.png")
    write_montage(
        rows,
        pathlib.Path(args.input_dir),
        pathlib.Path(args.output_dir),
        results_dir / "montage.png",
    )
    print(f"Report written to {results_dir}/ (summary.md, speedup.png, montage.png)")


if __name__ == "__main__":
    main()
