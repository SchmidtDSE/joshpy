#!/bin/bash
# link_targets.sh — symlink committed k8s batch target profiles into joshpy's
# targets dir (~/.josh/targets), so `joshpy.targets.load_target`/`list_targets`
# find them and `with_batch_remote("<name>")` works. Run by postCreateCommand;
# rerun by hand after adding a profile.
#
# A symlink (not a copy) keeps the committed .devcontainer/targets/<name>.json the
# single source of truth — edit it and the change is live, no re-copy. REPO_ROOT
# defaults to this checkout (set it to use another clone).
#
# Profiles hold no secrets: MinIO keys come from MINIO_ACCESS_KEY and
# MINIO_SECRET_KEY, and the kubectl context from gcloud credentials.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
_src="$REPO_ROOT/.devcontainer/targets"
_dest="$HOME/.josh/targets"
if [ -d "$_src" ]; then
    mkdir -p "$_dest"
    _n=0
    for _f in "$_src"/*.json; do
        [ -e "$_f" ] || continue                     # no profiles: skip (no nullglob needed)
        ln -sfn "$_f" "$_dest/$(basename "$_f")"      # -f replaces a stale copy/symlink
        echo "  target: $(basename "$_f" .json) -> $_f"
        _n=$(( _n + 1 ))
    done
    echo "  linked $_n target profile(s) into $_dest"
else
    echo "  targets: none committed at $_src — skipped"
fi
unset _src _dest _n _f
