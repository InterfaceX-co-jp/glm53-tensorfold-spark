#!/usr/bin/env bash
# Mechanical rebase probe: apply patches/*.patch (name order) on v0.3.4 (2f8e514) and on 0.3.6.2 (71377a5).
# Read-only for the project except the log written next to this script. Uses scratch worktrees of
# ~/.cache/tf-upstream and removes them at the end.
#
# Stage 1 (base, 2f8e514): mimic docker/Dockerfile: `git apply` then `patch -p1 --forward` fallback; commit each
#   patch. Besides verifying the series, this puts every intermediate blob in the shared object store, so the
#   stage-2 `git apply --3way` can find each patch's preimage blobs.
# Stage 2 (probe, 71377a5): per patch
#   clean   = `git apply --check` passes (then applied + committed)
#   3way    = `git apply --3way` succeeds with no conflicted paths (committed)
#   CONFLICT= 3way leaves conflicts or refuses; we record conflicted files + conflict-marker count, then reset and
#             measure `git apply --reject` per-file rejected hunks / total hunks, reset again and continue.
# A failed patch is not applied, so later patches that build on it may fail for that reason too (cascade).
set -u
UP=${UP:-$HOME/.cache/tf-upstream}
PROJ=${PROJ:-$HOME/glm53-tensorfold-spark}
BASE_SHA=2f8e514
NEW_SHA=71377a5
BASE_WT=$HOME/.cache/tf-rebase-base
PROBE_WT=$HOME/.cache/tf-rebase-probe
LOG=${LOG:-$PROJ/results/upstream-0362/rebase-probe.txt}
export GIT_AUTHOR_NAME=probe GIT_AUTHOR_EMAIL=probe@localhost GIT_COMMITTER_NAME=probe GIT_COMMITTER_EMAIL=probe@localhost

cleanup() {
  git -C "$UP" worktree remove --force "$BASE_WT" 2>/dev/null
  git -C "$UP" worktree remove --force "$PROBE_WT" 2>/dev/null
  git -C "$UP" worktree prune
}
cleanup
git -C "$UP" worktree add --detach "$BASE_WT" "$BASE_SHA" >/dev/null 2>&1 || exit 1
git -C "$UP" worktree add --detach "$PROBE_WT" "$NEW_SHA" >/dev/null 2>&1 || exit 1

exec > >(tee "$LOG") 2>&1
echo "# rebase probe $(date -Is)"
echo "# upstream clone $UP ; base $BASE_SHA ; target $NEW_SHA ; patches $PROJ/patches"
echo

hunks_total() { grep -c '^@@ ' "$1"; }

echo "## Stage 1: series on $BASE_SHA (Dockerfile method: git apply || patch -p1 --forward)"
BASE_FAIL=()
for p in "$PROJ"/patches/*.patch; do
  n=$(basename "$p")
  cd "$BASE_WT"
  if git apply "$p" 2>/tmp/probe.err; then how="git-apply"
  elif patch -p1 --forward --no-backup-if-mismatch -s < "$p" >/tmp/probe.err 2>&1; then how="patch-p1"
  else how="FAIL"; BASE_FAIL+=("$n"); git reset -q --hard; git clean -qfd; fi
  fmt=$(head -1 "$p" | grep -q '^From ' && echo format-patch || echo plain-diff)
  pre=$(grep -m1 '^diff --git' "$p" | awk '{print substr($3,1,2)}')
  if [ "$how" != FAIL ]; then git add -A && git commit -qm "$n" --no-verify; fi
  printf '%-40s %-10s %-12s prefix=%s hunks=%s\n' "$n" "$how" "$fmt" "$pre" "$(hunks_total "$p")"
  [ "$how" = FAIL ] && sed 's/^/    /' /tmp/probe.err | head -20
done
echo
echo "Stage-1 failures: ${BASE_FAIL[*]:-none}"
echo

echo "## Stage 2: each patch on $NEW_SHA with git apply --3way (committed on success, reset on conflict)"
declare -A RESULT
for p in "$PROJ"/patches/*.patch; do
  n=$(basename "$p")
  cd "$PROBE_WT"
  skip=0
  for f in "${BASE_FAIL[@]:-}"; do [ "$f" = "$n" ] && skip=1; done
  if [ $skip = 1 ]; then echo "=== $n : SKIPPED (fails on $BASE_SHA)"; RESULT[$n]=skipped; continue; fi
  if git apply --check "$p" 2>/dev/null; then
    git apply "$p"; git add -A; git commit -qm "$n" --no-verify
    echo "=== $n : clean"; RESULT[$n]=clean; continue
  fi
  git apply --3way "$p" >/tmp/probe.3w 2>&1; rc=$?
  unmerged=$(git diff --name-only --diff-filter=U)
  if [ $rc -eq 0 ] && [ -z "$unmerged" ]; then
    git add -A; git commit -qm "$n (3way)" --no-verify
    echo "=== $n : 3way"; RESULT[$n]=3way; continue
  fi
  echo "=== $n : CONFLICT (3way rc=$rc)"
  markers=0
  if [ -n "$unmerged" ]; then
    for f in $unmerged; do
      c=$(grep -c '^<<<<<<< ' "$f" 2>/dev/null || true); markers=$((markers + ${c:-0}))
      echo "    3way-conflict: $f ($c conflict regions)"
    done
  fi
  grep -E '^error:|does not exist|already exists|patch does not apply|with conflicts|Falling back' /tmp/probe.3w | sort -u | sed 's/^/    /' | head -30
  git reset -q --hard; git clean -qfd
  # per-file rejected hunks (direct apply, no 3-way): how much has to be hand-ported
  git apply --reject "$p" >/tmp/probe.rej 2>&1
  rejtot=0; ftot=0
  for r in $(find . -name '*.rej' -not -path './.git/*' | sort); do
    k=$(grep -c '^@@ ' "$r"); rejtot=$((rejtot + k)); ftot=$((ftot + 1))
    echo "    rejected: ${r#./} : $k hunk(s)"
  done
  grep -E "^error: .*(No such file|does not exist|already exists)" /tmp/probe.rej | sort -u | sed 's/^/    missing\/exists: /'
  echo "    summary: 3way-conflict-regions=$markers rejected-hunks=$rejtot/$(hunks_total "$p") in $ftot file(s)"
  RESULT[$n]="CONFLICT markers=$markers rej=$rejtot/$(hunks_total "$p")"
  git reset -q --hard; git clean -qfd
done
echo
echo "## Stage 2 summary (isolated: a failed patch is reset, so later patches cascade-fail on files it creates)"
for p in "$PROJ"/patches/*.patch; do n=$(basename "$p"); printf '%-40s %s\n' "$n" "${RESULT[$n]}"; done
echo

# Stage 3: keep-going. Apply each patch with --reject, keep every hunk that applies (drop .rej), commit, next.
# Files missing on 71377a5 are retried with --exclude so the rest of the patch lands. Rejected hunks are split
# into UP (file exists upstream at 2f8e514: a real clash with upstream or with a hunk of ours that was dropped)
# and OWN (file created by our own series: only a cascade of an earlier dropped hunk).
echo "## Stage 3: keep-going on $NEW_SHA (git apply --reject; partial result committed; rejects classified)"
cd "$PROBE_WT"; git reset -q --hard "$NEW_SHA"; git clean -qfd
declare -A R3
for p in "$PROJ"/patches/*.patch; do
  n=$(basename "$p")
  excl=()
  for _ in 1 2 3 4; do
    git apply --reject "${excl[@]}" "$p" >/tmp/probe.k 2>&1
    miss=$(grep -oE "^error: [^:]+: (No such file or directory|does not exist in index|already exists in working directory)" /tmp/probe.k | sed -E 's/^error: ([^:]+): .*/\1/' | sort -u)
    [ -z "$miss" ] && break
    git reset -q --hard; git clean -qfd
    for m in $miss; do excl+=("--exclude=$m"); done
  done
  up=0; own=0; lines=""
  for r in $(find . -name '*.rej' -not -path './.git/*' | sort); do
    f=${r#./}; f=${f%.rej}; k=$(grep -c '^@@ ' "$r")
    if git cat-file -e "$BASE_SHA:$f" 2>/dev/null; then up=$((up + k)); t=UP; else own=$((own + k)); t=OWN; fi
    lines+="    rej $t $f : $k"$'\n'
  done
  for e in "${excl[@]:-}"; do [ -n "$e" ] && lines+="    skipped-file ${e#--exclude=} (missing or already exists on target)"$'\n'; done
  find . -name '*.rej' -not -path './.git/*' -delete
  git add -A; git commit -qm "$n (keep-going)" --no-verify --allow-empty
  tot=$(hunks_total "$p")
  if [ $up -eq 0 ] && [ $own -eq 0 ] && [ ${#excl[@]} -eq 0 ]; then st=clean; else st="rejUP=$up rejOWN=$own skippedFiles=${#excl[@]} of $tot hunks"; fi
  R3[$n]=$st
  echo "=== $n : $st"; printf '%s' "$lines"
done
echo
echo "## Stage 3 summary"
for p in "$PROJ"/patches/*.patch; do n=$(basename "$p"); printf '%-40s %s\n' "$n" "${R3[$n]}"; done
echo

# Stage 4: the whole series as one squash, 3-way merged onto 71377a5 (rename detection on: follows
# glm5_next/cuda/comm.py -> cuda/comm.py). Lists every conflicted path and its conflict-region count.
echo "## Stage 4: squashed series (2f8e514..series tip) merged onto $NEW_SHA with git merge-tree (renames followed)"
TIP=$(git -C "$BASE_WT" rev-parse HEAD)
out=$(git -C "$UP" merge-tree --write-tree --name-only --merge-base="$BASE_SHA" "$NEW_SHA" "$TIP")
tree=$(printf '%s\n' "$out" | head -1)
printf '%s\n' "$out" | sed -n '2,/^$/p' | sed '/^$/d' | sort -u > /tmp/probe.conf
echo "conflicted paths: $(wc -l < /tmp/probe.conf)"
while read -r f; do
  c=$(git -C "$UP" cat-file -p "$tree:$f" 2>/dev/null | grep -c '^<<<<<<< ' || true)
  printf '    %-70s %s region(s)\n' "$f" "${c:-?}"
done < /tmp/probe.conf
echo "messages:"
printf '%s\n' "$out" | sed -n '/^$/,$p' | grep -E "CONFLICT|renamed|Auto-merging" | grep -v "^Auto-merging" | sed 's/^/    /' | head -80
cd /
cleanup
echo "# worktrees removed"
