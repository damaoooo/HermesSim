#!/usr/bin/env bash
# One-time setup on a Shaheen login node (compute nodes have no internet):
#   uv environment (.venv), a private JDK 21 (.jdk/), and bin/gsat-1.0-opt.jar.
# usage: bash scripts/shaheen/setup.sh        (from the repository root)
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --locked

if [ ! -x .jdk/bin/java ]; then
    url=https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse
    mkdir -p .jdk
    curl -fL --retry 3 "$url" | tar -xz -C .jdk --strip-components=1
fi
.jdk/bin/java -version

JAVA_HOME=$PWD/.jdk gsat/build.sh
