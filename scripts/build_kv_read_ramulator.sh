#!/usr/bin/env bash
set -euo pipefail

readonly KV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly KV_PROJECT_ROOT="$(cd -- "${KV_SCRIPT_DIR}/.." && pwd)"
readonly KV_RAMULATOR_ROOT="${KV_PROJECT_ROOT}/third_party/ramulator2"
readonly KV_BUILD_DIR="${RAMULATOR_BUILD_DIR:-${KV_PROJECT_ROOT}/build/ramulator2}"

# Keep the frozen KV READ baseline in the original Ramulator tree.
"${KV_SCRIPT_DIR}/build_sieve_ramulator.sh"
install -m 0644 \
    "${KV_PROJECT_ROOT}/ramulator/extensions/sieve_kv_read/source/sieve_kv_read_frontend.cpp" \
    "${KV_RAMULATOR_ROOT}/src/ramulator/frontend/impl/memory_trace/sieve_kv_read_frontend.cpp"
python3 - "${KV_RAMULATOR_ROOT}/src/ramulator/frontend/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
entry = "  impl/memory_trace/sieve_kv_read_frontend.cpp\n"
anchor = "  impl/memory_trace/sieve_mixed_frontend.cpp\n"
if entry not in text:
    if text.count(anchor) != 1:
        raise SystemExit("expected exactly one installed SieveMixed source entry")
    path.write_text(text.replace(anchor, anchor + entry))
PY
cmake -S "${KV_RAMULATOR_ROOT}" -B "${KV_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release -DRAMULATOR_PYTHON_BINDINGS=ON
cmake --build "${KV_BUILD_DIR}" --parallel "${RAMULATOR_BUILD_JOBS:-2}"
