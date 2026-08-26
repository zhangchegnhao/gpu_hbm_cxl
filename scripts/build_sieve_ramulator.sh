#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly RAMULATOR_ROOT="${PROJECT_ROOT}/third_party/ramulator2"
readonly EXTENSION_ROOT="${PROJECT_ROOT}/ramulator/extensions/sieve_hbm_pim"
readonly PATCH_V0="${EXTENSION_ROOT}/patches/ramulator2-cycle-v0.patch"
readonly PATCH_V1="${EXTENSION_ROOT}/patches/ramulator2-cycle-v1.patch"
readonly BUILD_DIR="${RAMULATOR_BUILD_DIR:-${PROJECT_ROOT}/build/ramulator2}"

"${SCRIPT_DIR}/fetch_ramulator2.sh"

patch_is_installed() {
    case "$(basename -- "$1")" in
        ramulator2-cycle-v0.patch)
            grep -Fq 'PIM_GWRITE = 2' "${RAMULATOR_ROOT}/src/ramulator/base/request.h" &&
                grep -Fq 'sieve_hbm_pim_controller.cpp' \
                    "${RAMULATOR_ROOT}/src/ramulator/controller/CMakeLists.txt" &&
                grep -Fq 'sieve_pim_frontend.cpp' \
                    "${RAMULATOR_ROOT}/src/ramulator/frontend/CMakeLists.txt"
            ;;
        ramulator2-cycle-v1.patch)
            grep -Fq 'sieve_hbm_pim_controller_v1.cpp' \
                "${RAMULATOR_ROOT}/src/ramulator/controller/CMakeLists.txt" &&
                grep -Fq 'sieve_mixed_frontend.cpp' \
                    "${RAMULATOR_ROOT}/src/ramulator/frontend/CMakeLists.txt"
            ;;
        *)
            return 1
            ;;
    esac
}

for patch in "${PATCH_V0}" "${PATCH_V1}"; do
    if git -C "${RAMULATOR_ROOT}" apply --check --ignore-space-change --ignore-whitespace "${patch}" 2>/dev/null; then
        git -C "${RAMULATOR_ROOT}" apply --ignore-space-change --ignore-whitespace "${patch}"
    elif patch_is_installed "${patch}"; then
        :
    else
        echo "error: Ramulator patch is neither applicable nor already applied: ${patch}" >&2
        exit 1
    fi
done

install -m 0644 \
    "${EXTENSION_ROOT}/source/sieve_pim_frontend.cpp" \
    "${RAMULATOR_ROOT}/src/ramulator/frontend/impl/memory_trace/sieve_pim_frontend.cpp"
install -m 0644 \
    "${EXTENSION_ROOT}/source/sieve_hbm_pim_controller.cpp" \
    "${RAMULATOR_ROOT}/src/ramulator/controller/impl/sieve_hbm_pim_controller.cpp"
install -m 0644 \
    "${EXTENSION_ROOT}/source/sieve_mixed_frontend.cpp" \
    "${RAMULATOR_ROOT}/src/ramulator/frontend/impl/memory_trace/sieve_mixed_frontend.cpp"
install -m 0644 \
    "${EXTENSION_ROOT}/source/sieve_hbm_pim_controller_v1.cpp" \
    "${RAMULATOR_ROOT}/src/ramulator/controller/impl/sieve_hbm_pim_controller_v1.cpp"

cmake -S "${RAMULATOR_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --parallel "${RAMULATOR_BUILD_JOBS:-2}"

echo "Built Sieve Ramulator extension in ${BUILD_DIR}"
