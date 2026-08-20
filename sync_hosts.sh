#!/usr/bin/env bash
# Sync ALL akew_* sources to every GPU host, then VERIFY each landed.
#
# Exists because partial syncing bit this project twice in one session: once
# the three-way router ran with reject_floor silently ignored (producing
# results that looked plausible and meant nothing), and once the 7B sweep
# crashed on a stale akew_router.py after a file was left out of a
# hand-written sync list. Both were hand-maintained per-file lists that
# drifted from what had actually changed. Syncing the whole glob removes the
# class of bug rather than the instance.
#
# The verification step is the point: scp reporting success does not prove the
# remote file contains the symbol the run needs, and a missing symbol either
# crashes late (best case, hours in) or is silently ignored (worst case).
#
# Usage: ./sync_hosts.sh [symbol_to_verify ...]
set -euo pipefail

HOSTS=(g6e4xlarge virginia-g6e2xlarge)
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src"
REMOTE_DIR="~/kw/cake_prototype/src"
VERIFY_SYMBOLS=("$@")

fail=0
for host in "${HOSTS[@]}"; do
  echo "=== $host ==="

  # Fail fast on a syntax error rather than shipping a broken file to a GPU
  # box and finding out after the model has loaded.
  for f in "$SRC_DIR"/akew_*.py; do
    python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$f" || {
      echo "  SYNTAX ERROR in $(basename "$f") -- nothing synced"; exit 1; }
  done

  if ! scp -o ConnectTimeout=10 "$SRC_DIR"/akew_*.py "$host:$REMOTE_DIR/" >/dev/null 2>&1; then
    echo "  SCP FAILED"; fail=1; continue
  fi

  local_count=$(ls -1 "$SRC_DIR"/akew_*.py | wc -l)
  remote_count=$(ssh -o ConnectTimeout=10 "$host" "ls -1 $REMOTE_DIR/akew_*.py | wc -l")
  echo "  files: local=$local_count remote=$remote_count"
  [ "$local_count" -le "$remote_count" ] || { echo "  COUNT MISMATCH"; fail=1; }

  for sym in "${VERIFY_SYMBOLS[@]:-}"; do
    [ -z "$sym" ] && continue
    n=$(ssh -o ConnectTimeout=10 "$host" "grep -l '$sym' $REMOTE_DIR/akew_*.py 2>/dev/null | wc -l")
    if [ "$n" -gt 0 ]; then
      echo "  verified symbol present: $sym (in $n file(s))"
    else
      echo "  MISSING SYMBOL: $sym"; fail=1
    fi
  done
done

if [ "$fail" -ne 0 ]; then
  echo "SYNC INCOMPLETE -- do not launch runs against these hosts"
  exit 1
fi
echo "sync OK on all hosts"
