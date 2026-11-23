#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime-env.sh
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
if [[ "${1:-}" == --reuse-local ]]; then
  # Reuse audited dependencies without modifying the existing checkout/venv.
  source_env="${2:-$HOME/projects/sglang/.venv}"
  uv venv --allow-existing --python "$source_env/bin/python" .venv
  source_site=$("$source_env/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  target_site=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  SOURCE_SITE="$source_site" TARGET_SITE="$target_site" .venv/bin/python - <<'PY'
import os
from pathlib import Path
Path(os.environ['TARGET_SITE'], 'local-sglang.pth').write_text(
    'import site; site.addsitedir(' + repr(os.environ['SOURCE_SITE']) + ')\n')
PY
  .venv/bin/python - <<'PY'
from pathlib import Path
import sys
entry = Path('.venv/bin/sglang')
entry.write_text('#!' + sys.executable + '\nfrom sglang.cli.main import main\nmain()\n')
entry.chmod(0o755)
PY
else
  uv venv --allow-existing --python 3.12 .venv
  uv pip install --python .venv/bin/python --prerelease=allow "sglang==${SGLANG_VERSION:-0.5.21}"
  uv pip install --python .venv/bin/python 'sgl-eval==0.1.3' 'antlr4-python3-runtime==4.9.3' 'socksio==1.0.0'
fi
mkdir -p results
.venv/bin/sglang serve --help > results/server-help.txt
.venv/bin/python scripts/environment.py --output results/environment.json
if ! .venv/bin/python -c 'from flashinfer import fp4_quantize; from flashinfer import mm_fp4; print("FP4 imports OK")'; then
  echo 'FP4 imports failed; stage 0 is incomplete.' >&2
  exit 1
fi
echo 'Help and FP4 imports passed. GPU GEMM/JIT and stage 1 still require a live GPU.'
