#!/usr/bin/env bash
set -euo pipefail
which mkdocs >/dev/null || (echo "Installe mkdocs: pip install mkdocs mkdocs-material" && exit 1)
mkdocs build
mkdocs gh-deploy --clean