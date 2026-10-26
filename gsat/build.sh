#!/usr/bin/env bash
# Build bin/gsat-1.0-opt.jar: the gsat release jar (v1.0.0, Ghidra 10.3 with the SOG modifications) with the
# com.gsat classes recompiled from gsat/src. Needs a JDK >= 17 (javac, jar) on PATH or in $JAVA_HOME.
# usage: gsat/build.sh            (run from the repository root)
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=bin/gsat-1.0.jar
OUT=bin/gsat-1.0-opt.jar
URL=https://github.com/sgfvamll/gsat/releases/download/v1.0.0/gsat-1.0.jar
SHA=ab7152a9bf9f608cfad848a9d719243ff816a218188dde185bd3cb29a1090c93
JBIN=${JAVA_HOME:+$JAVA_HOME/bin/}

mkdir -p bin
if [ ! -f "$BASE" ]; then
    curl -fL --retry 3 -o "$BASE.tmp" "$URL"
    mv "$BASE.tmp" "$BASE"
fi
echo "$SHA  $BASE" | sha256sum -c -

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# -implicit:none / empty -sourcepath: compile only com.gsat; the release jar also ships Ghidra's .java files.
"${JBIN}javac" --release 17 -nowarn -encoding UTF-8 -implicit:none -sourcepath "" -cp "$BASE" -d "$tmp" \
    $(find gsat/src/main/java -name "*.java") 2>&1 | grep -v "^Note:" || true
cp "$BASE" "$OUT.tmp"
(cd "$tmp" && "${JBIN}jar" uf "$OLDPWD/$OUT.tmp" com)
mv "$OUT.tmp" "$OUT"
echo "built $OUT ($(find "$tmp" -name '*.class' | wc -l) classes)"
