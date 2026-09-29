# Multi-session snapshot cache (M2) on top of the latent KV cache (0060): design sketch

Status: implemented as `patches/0110-glm-session-cache.patch` (`GLM53_TF_SESSION_GIB`, `families/glm5_next/cuda/
sessions.py`; details in `docs/PATCHES.md`). What changed from this sketch: restores COPY an entry's pages into the
live caches (a few ms at 30k tokens) instead of paging the kernels' addresses (none of the kernels changed); page
keys include the token after the page (MTP rows) and, for fast prefill, the whole chunk; the model layers' index keys
and gates are not stored (0065's ring: the tail rides in `conv`); snapshots also keep the DFlash2 drafter's window;
checkpoints are taken at fork points and every 16k tokens ("marks"); batching (0030) keeps its own slots, except with
patches/0180 (`GLM53_TF_BATCH_SESSIONS=1`: one store behind every 0120 slot, restores copied into a free slot). The sketch
below is kept as written. The optional NVMe spill of section 3 is patches/0250 (`GLM53_TF_SESSION_DISK`, `sessdisk.py`): write-through or
on eviction, restored straight into the live slot, persistent across restarts.

Written 2026-09-27 alongside `patches/0060-glm-latent-kv.patch`
(`GLM53_TF_LATENT_KV=1`), which makes the attention cache small enough to keep many conversations resident.
Numbers are per rank unless stated, for GLM-5.3-Flash on 2 x DGX Spark (TP = 2).

## 1. Why now

Today one sequence owns the caches, and `GlmEngine.cache` keeps snapshots of only the last prompt and reply. An
agent that switches between sessions (subagents, parallel tool loops, several users) re-prefills the whole context
on every switch: 30k tokens at ~420 tok/s is ~70 s of TTFT. The per-token attention cache was the blocker:

| Per token, per rank | Expanded (0060 off) | Latent (0060 on) | Latent, fp8 (future) |
| --- | ---: | ---: | ---: |
| DSA + MTP attention cache (12 layers) | 384 KB | 12 KB | 6 KB |
| Indexer caches (keys, gates, 1/4 pool key; 12 layers) | 6.75 KB | 6.75 KB | ~3.5 KB |
| **Total** | **390.75 KB** | **18.75 KB** | **~9.5 KB** |

A session's fixed state does not depend on its length:

| Fixed state per session snapshot | Size |
| --- | ---: |
| KDA recurrent state: 34 layers x 32 local heads x 128 x 128 x fp32 | 71.3 MB |
| Conv windows: 34 x 3 x 12,288 x bf16 | 2.5 MB |
| Pending MTP input rows, sampler/drafter bookkeeping | < 0.1 MB |
| **Snapshot** | **~74 MB** |

## 2. Memory math (35 GB of free memory a rank for sessions)

Size of one resident session = tokens x 18.75 KB + snapshots x 74 MB (one snapshot at the last turn end, plus
the live state of the running session).

| Session length | Latent KV | + 1 snapshot | Sessions in 35 GB | Expanded KV (0060 off) | Sessions in 35 GB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8k | 0.15 GB | 0.22 GB | ~155 | 3.1 GB | 11 |
| 30k | 0.56 GB | 0.64 GB | ~55 | 11.7 GB | 2-3 |
| 100k | 1.88 GB | 1.95 GB | ~17 | 39 GB | 0 |
| 1M | 18.8 GB | 18.9 GB | 1 (+ ~25 x 30k beside it) | 391 GB | 0 |

Snapshots every 4k tokens (to resume at arbitrary earlier prefixes) cost 74 MB each, more than the 4k tokens of
latent KV they cover (75 MB). Keep them sparse: one at each turn end, plus at most one mid-session checkpoint per
~16k tokens. Q1 (3-bit experts) would add ~19 GB a rank to this pool; an fp8 latent would halve the per-token
column (a new configuration with its own quality gate).

Other per-token memory to watch at long contexts: the DFlash2 drafter keeps its own context K/V
(`Drafter.kc/vc`, sized by the capacity), and 0050's `LongScratch` scores `rows x capacity / 4` pools (24 MB at
1M). The drafter cache should become a ring buffer over its sliding window before sessions are sized by the
store's capacity.

## 3. Mechanism

**Paged latent store.** One pool per rank of fixed pages, allocated at start:

- a page holds 256 consecutive positions of one session: every DSA layer's latent rows, the MTP head's latent
  rows, and the indexer's keys, gates and 64 pool keys (4.8 MB a page at 18.75 KB a token);
- a session owns a page table (logical page -> physical page), one int32 per 256 tokens;
- the kernels address a latent row as `PT[t // 256] * 256 + t % 256`: the dense and sparse latent kernels, the
  latent write, the indexer write, pool-key and scoring kernels. Addresses change no arithmetic (the chunk and
  tile order is by logical position), so paging keeps every bit; the page table is a device tensor, so graphs
  still capture (the table's contents change, not its address).

**Prefix sharing.** Full pages are immutable once committed. A radix tree keyed by each full page's token ids
(chained hash) maps prefixes to physical pages with a reference count. A new or resumed session maps the longest
shared prefix of full pages and copies the partial last page (copy-on-write). A shared system prompt costs its
pages once. The bits are identical to a private prefill, because a latent row depends only on its token and its
prefix.

**KDA state snapshots.** The recurrent state cannot be rebuilt from the KV cache, so resuming at position p needs
the state at a snapshot point s <= p, then a prefill of tokens s .. p. Snapshots are taken:

- at each turn end (the prompt + reply state the engine already keeps as `decode.Snapshot`);
- optionally at a canonical checkpoint every 16k tokens of a long prompt, so resuming inside a long shared
  document needs at most 16k tokens of prefill.

A snapshot is `(rec, conv, pending MTP rows, mtp_len, drafter_end)` plus the session's page table and length.
Taking or restoring one is a device copy of 74 MB (~0.5 ms).

**Eviction.** LRU over sessions, costed by bytes freed. Pages go when their reference count reaches zero.
Snapshots go before their pages (a session without a snapshot can still share its prefix pages with a later
fresh prefill of the same tokens, which re-runs only the KDA chain). Optional spill to NVMe: a 30k session is
0.64 GB, ~0.15 s at 4-5 GB/s. On GB10 the CPU and GPU share one LPDDR5x pool, so there is no cheaper "host"
tier.

**Two ranks.** The latent is replicated (both ranks hold the full 512-wide rows), and each rank holds its heads'
KDA state. Rank 0 decides admissions, page allocations and evictions and shares them before each round
(`GlmEngine._share`, as patches/0030 does for slots). Both ranks then run identical allocators on identical
inputs, so page tables match without further traffic.

**Batching (0030).** A batch slot becomes a session handle: `State.kc[i]` becomes a view through the session's
page table, and `Batcher._place` already chooses the slot whose kept state resumes the longest prefix. That
choice becomes a radix-tree lookup across every resident session, not just the N slots.

## 4. Exactness

No new arithmetic. A resumed session's state equals a fresh prefill's because:

1. latent rows and indexer rows are row-local functions of the prefix, and prefill chunk size changes no bit
   (tested for 0060: 64- vs 512-row chunks give identical caches);
2. KDA snapshots are bit copies of the state a fresh prefill reaches at the same position;
3. paging changes addresses, not the order of any reduction.

The existing tests carry over with a session store underneath: resumed == fresh, drafted == serial, batched ==
alone. One new test: two sessions sharing a prefix, interleaved, each equal to its own serial run from scratch.

## 5. Gain and cost

- A switch back to a cached 30k session re-prefills only the new turn (typically 0.5-5k tokens): 1-12 s at
  420 tok/s (0.3-3 s after P2), instead of ~70 s. The first token after a pure resume costs one forward.
- A shared 10k-token system prompt across 20 agents: 0.19 GB once instead of 20 x 0.19 GB, and 24 s of prefill
  once instead of 20 times.
- Effort after 0060: ~5 days. Paged addressing in the latent, indexer and sparse kernels (1.5 d); page allocator,
  radix tree and eviction (1.5 d); snapshot store and the engine/batch integration (1 d); tests (1 d).
- Risk: low for exactness (no arithmetic change). The allocator's determinism across ranks is the main
  correctness risk; it is covered by sharing every decision from rank 0.
