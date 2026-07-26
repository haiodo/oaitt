#!/bin/bash
#
# Run the full OAITT ASR benchmark suite and generate an HTML report.
#
# Usage:
#   ./benchmark.sh                      # full mode (whole sample file)
#   ./benchmark.sh --mode short         # 20s audio, 5 iterations
#   ./benchmark.sh --mode long -i 5     # 60s audio, 5 iterations
#   ./benchmark.sh -s run_gigaam_mlx_asr.sh   # only one engine
#
# Any extra arguments are passed through to tests/test_benchmark.py.
# Report paths can be overridden via OUT_DIR / OPEN_REPORT env vars.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/venv/bin/activate"
else
    echo "venv not found at $SCRIPT_DIR/venv - run ./prepare.sh first" >&2
    exit 1
fi

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/bench_results}"
STAMP="$(date +%Y%m%d_%H%M%S)"
HTML_OUT="$OUT_DIR/benchmark_${STAMP}.html"
JSON_OUT="$OUT_DIR/benchmark_${STAMP}.json"
mkdir -p "$OUT_DIR"

# Free the server port before starting - a stale server would benchmark the wrong engine.
PORT="${PORT:-9007}"
STALE_PID="$(lsof -ti ":$PORT" 2>/dev/null || true)"
if [ -n "$STALE_PID" ]; then
    echo "Killing stale process(es) on port $PORT: $STALE_PID"
    kill -9 $STALE_PID 2>/dev/null || true
    sleep 2
fi

MODE_ARGS=""
if [[ ! " $* " =~ " --mode " ]] && [[ ! " $* " =~ " -m " ]]; then
    MODE_ARGS="--mode full"
fi

python -u -m tests.test_benchmark \
    $MODE_ARGS \
    --json "$JSON_OUT" \
    --html "$HTML_OUT" \
    "$@"

echo ""
echo "Report: $HTML_OUT"
ln -sf "$(basename "$HTML_OUT")" "$OUT_DIR/latest.html"

if [ "${OPEN_REPORT:-1}" = "1" ] && command -v open >/dev/null 2>&1; then
    open "$HTML_OUT"
fi
