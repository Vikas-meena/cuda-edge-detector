#!/usr/bin/env bash
# Builds the detector, runs it on data/input and collects proof-of-execution
# artifacts in results/.
#
# Usage: ./run.sh [edge_detector options]
#   Any arguments are passed straight to bin/edge_detector, for example
#   ./run.sh --sigma=2.0 --block_size=32
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! ls data/input/*.png >/dev/null 2>&1; then
  echo "data/input is empty, downloading the USC-SIPI test images ..."
  ./scripts/fetch_data.sh
fi

make build

mkdir -p results data/output
./bin/edge_detector "$@" | tee results/run_log.txt

# The report needs Pillow and matplotlib; the run itself does not.
if python3 -c "import PIL, matplotlib" >/dev/null 2>&1; then
  python3 scripts/make_report.py
else
  echo "Skipping the report: python3 with Pillow and matplotlib not found."
fi
