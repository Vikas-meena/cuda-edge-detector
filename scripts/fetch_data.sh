#!/usr/bin/env bash
# Downloads test images from the USC-SIPI Image Database and converts them
# from TIFF to PNG (the detector reads PNG/JPEG/BMP/PGM/PPM, not TIFF).
#
# Usage: scripts/fetch_data.sh [volume ...]
#   Volumes default to "misc aerials". Each one is a zip file published at
#   https://sipi.usc.edu/database/ .
#
# Requires: curl, unzip, python3 with Pillow (pip install pillow).
set -euo pipefail

readonly BASE_URL="https://sipi.usc.edu/database"
readonly ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly INPUT_DIR="${ROOT_DIR}/data/input"

volumes=("$@")
if [[ ${#volumes[@]} -eq 0 ]]; then
  volumes=(misc aerials)
fi

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT
mkdir -p "${INPUT_DIR}"

for volume in "${volumes[@]}"; do
  echo "Downloading ${volume}.zip ..."
  curl -fsSL -o "${work_dir}/${volume}.zip" "${BASE_URL}/${volume}.zip"
  unzip -q -o "${work_dir}/${volume}.zip" -d "${work_dir}"
done

echo "Converting TIFF to PNG ..."
python3 - "${work_dir}" "${INPUT_DIR}" <<'PYTHON'
import pathlib
import sys

from PIL import Image

source, target = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
count = 0
for path in sorted(source.rglob("*.tiff")):
    volume = path.parent.name
    name = f"{volume}_{path.stem}.png"
    with Image.open(path) as image:
        mode = "L" if image.mode in ("1", "L", "I;16", "I") else "RGB"
        image.convert(mode).save(target / name)
    count += 1
print(f"Wrote {count} PNG files to {target}")
PYTHON
