#!/usr/bin/env bash
set -euo pipefail

readonly CXL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly CXL_PROJECT_ROOT="$(cd -- "${CXL_SCRIPT_DIR}/.." && pwd)"
readonly CXL_BASE_ROOT="${CXL_PROJECT_ROOT}/third_party/ramulator2"
readonly CXL_RAMULATOR_ROOT="${CXL_RAMULATOR_ROOT:-${CXL_PROJECT_ROOT}/third_party/ramulator2_cxl}"
readonly CXL_BUILD_DIR="${CXL_RAMULATOR_BUILD_DIR:-${CXL_PROJECT_ROOT}/build/ramulator2_cxl}"
readonly CXL_EXTENSION_ROOT="${CXL_PROJECT_ROOT}/ramulator/extensions/sieve_cxl"

# Build the frozen baseline first; CXL is compiled in a separate ignored tree.
"${CXL_SCRIPT_DIR}/build_kv_read_ramulator.sh"
if [[ ! -d "${CXL_RAMULATOR_ROOT}/.git" ]]; then
    cp -a "${CXL_BASE_ROOT}" "${CXL_RAMULATOR_ROOT}"
fi
if [[ "$(git -C "${CXL_RAMULATOR_ROOT}" rev-parse HEAD)" != "$(git -C "${CXL_BASE_ROOT}" rev-parse HEAD)" ]]; then
    echo "error: CXL Ramulator tree is at a different revision" >&2
    exit 1
fi

install -m 0644 \
    "${CXL_EXTENSION_ROOT}/source/sieve_cxl_kv_frontend.cpp" \
    "${CXL_RAMULATOR_ROOT}/src/ramulator/frontend/impl/memory_trace/sieve_cxl_kv_frontend.cpp"
install -m 0644 \
    "${CXL_EXTENSION_ROOT}/source/sieve_cxl_memory_controller.cpp" \
    "${CXL_RAMULATOR_ROOT}/src/ramulator/controller/impl/sieve_cxl_memory_controller.cpp"
python3 - "${CXL_RAMULATOR_ROOT}/src/ramulator/frontend/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
entry = "  impl/memory_trace/sieve_cxl_kv_frontend.cpp\n"
anchor = "  impl/memory_trace/sieve_kv_read_frontend.cpp\n"
if entry not in text:
    if text.count(anchor) != 1:
        raise SystemExit("expected exactly one installed SieveKVRead source entry")
    path.write_text(text.replace(anchor, anchor + entry))
PY
python3 - "${CXL_RAMULATOR_ROOT}/src/ramulator/controller/CMakeLists.txt" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
entry = "  impl/sieve_cxl_memory_controller.cpp\n"
anchor = "  impl/sieve_hbm_pim_controller_v1.cpp\n"
if entry not in text:
    if text.count(anchor) != 1:
        raise SystemExit("expected exactly one installed SieveHBMPIMV1 source entry")
    path.write_text(text.replace(anchor, anchor + entry))
PY
cmake -S "${CXL_RAMULATOR_ROOT}" -B "${CXL_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release -DRAMULATOR_PYTHON_BINDINGS=ON \
    -DFETCHCONTENT_SOURCE_DIR_FMT="${CXL_RAMULATOR_ROOT}/ext/fmt" \
    -DFETCHCONTENT_SOURCE_DIR_YAML-CPP="${CXL_RAMULATOR_ROOT}/ext/yaml-cpp" \
    -DFETCHCONTENT_SOURCE_DIR_NANOBIND="${CXL_RAMULATOR_ROOT}/ext/nanobind"
cmake --build "${CXL_BUILD_DIR}" --parallel "${RAMULATOR_BUILD_JOBS:-2}"

echo "Built CXL Sieve Ramulator extension in ${CXL_BUILD_DIR}"
