#!/bin/bash
#
# Shared venv guard for the run_*.sh scripts.
#
# Source this, then call require_venv <venv-dir-name> <prepare-script> <import-check>:
#
#   source "${SCRIPT_DIR}/scripts/venv_guard.sh"
#   require_venv venv ./prepare-gigaam.sh gigaam
#
# Sets PYTHON to the interpreter of that venv. Exits with instructions if the
# venv is missing or the engine's package is not installed in it.
#

require_venv() {
    local venv_name="$1"
    local prepare_script="$2"
    local import_check="$3"

    local venv_dir="${PROJECT_ROOT}/${venv_name}"
    PYTHON="${venv_dir}/bin/python"

    if [ ! -x "$PYTHON" ]; then
        echo "Error: virtualenv '${venv_name}' not found at ${venv_dir}" >&2
        echo "" >&2
        echo "Set it up first:" >&2
        echo "    ${prepare_script}" >&2
        exit 1
    fi

    if [ -n "$import_check" ] && ! "$PYTHON" -c "import ${import_check}" 2>/dev/null; then
        echo "Error: package '${import_check}' is not installed in ${venv_name}" >&2
        echo "" >&2
        echo "Reinstall dependencies:" >&2
        echo "    ${prepare_script}" >&2
        exit 1
    fi

    export PYTHON
}
