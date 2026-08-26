#!/usr/bin/env bash
set -euo pipefail

readonly RAMULATOR_URL="https://github.com/CMU-SAFARI/ramulator2.git"
readonly RAMULATOR_COMMIT="b30320bc9385b708e86b67ebb9f48858cc66d798"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly DESTINATION="${PROJECT_ROOT}/third_party/ramulator2"

if [[ -e "${DESTINATION}" ]]; then
    actual_commit="$(git -C "${DESTINATION}" rev-parse HEAD 2>/dev/null || true)"
    if [[ "${actual_commit}" == "${RAMULATOR_COMMIT}" ]]; then
        echo "Ramulator 2.1 is already present at ${RAMULATOR_COMMIT}"
        exit 0
    fi
    if [[ -d "${DESTINATION}/.git" && -z "${actual_commit}" ]]; then
        git -C "${DESTINATION}" fetch --depth 1 origin "${RAMULATOR_COMMIT}"
        git -C "${DESTINATION}" checkout --detach FETCH_HEAD
    else
        echo "error: ${DESTINATION} exists at unexpected revision ${actual_commit:-unknown}" >&2
        exit 1
    fi
else
    git clone --filter=blob:none --no-checkout "${RAMULATOR_URL}" "${DESTINATION}"
    git -C "${DESTINATION}" fetch --depth 1 origin "${RAMULATOR_COMMIT}"
    git -C "${DESTINATION}" checkout --detach FETCH_HEAD
fi
actual_commit="$(git -C "${DESTINATION}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${RAMULATOR_COMMIT}" ]]; then
    echo "error: failed to check out the pinned Ramulator revision" >&2
    exit 1
fi
echo "Fetched Ramulator 2.1 at ${RAMULATOR_COMMIT}"
