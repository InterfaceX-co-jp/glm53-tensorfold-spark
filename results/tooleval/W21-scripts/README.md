# W21 tool-calling runs: what is published, and helper scripts

Write-up: [`docs/RESULTS.md`](../../../docs/RESULTS.md) W21 and [`docs/TOOL-CALLING.md`](../../../docs/TOOL-CALLING.md) §7.
All runs: production image b11, client on another machine over an SSH port forward (the `<port>` in commands and
reports is that forward), `scripts/tooleval/run.sh`, tool-eval-bench c7b5b95, spark-bench 125ba16.

| dir | run |
| --- | --- |
| `20260930-2135-sb-off-fixes/` | F1 spark-bench: fixes on, thinking off (F1's tool-eval-bench run is `20260930-teb-off-fixes-PARTIAL/`) |
| `20260930-2201-both-high-fixes/` | F2 both benches: fixes on, thinking high |
| `20261001-0015-both-off-baseline/` | B1 both benches: fixes off (`GLM53_TF_TOOL_FIXES` unset), thinking off |
| `20261001-0049-chains-low-fixes/` | F3 chains: tool-eval-bench category C + spark-bench agentic x3, fixes on, thinking low |
| `20261001-00*-teb-*-C-t03-s{1,2,3}/` | category C at temperature 0.3, `--seed 1/2/3`: fixes off, fixes on thinking off, fixes on high |
| `20261001-0100-rr01-probe-fixes/` | RR-01 probe (`rr01_probe.py`): 10 seeds each with the null assistant content sent as `null` and as `"None"` |

Published from each run: the command line (`*.cmd`), `/health` and `/v1/models` before / after, spark-bench's report
(`sb/runs/*.md`) and CSV (`sb/spark_bench.csv`), and tool-eval-bench's JSON and Markdown report. Edits and omissions:

- Local paths are repo-relative, the forward's port is `<port>`, and the client's host name and platform are
  `<client>`.
- The two full 69-scenario tool-eval-bench runs (B1, F2) are published as `teb-summary.json` (as for the W20 run): the
  bench's JSON with each scenario's raw log, per-turn timings and the local host fields removed, and one fixture e-mail
  domain shown as `<redacted>`. Their Markdown reports (full transcripts of the bench's fixture data) are omitted. The
  category-C runs keep the full JSON and report.
- spark-bench artifacts: only the two that the write-up relies on are kept (RR-01 transcripts and `VIS-05.html` for
  B1, F1, F2). The other transcripts contain the bench's fixture e-mails and credentials, which trip
  `scripts/check-public.sh`.
- Not published: run logs, spark-bench's HTML reports (same content as the `.md`) and the raw request dumps
  (`sb-dump/raw_dump.jsonl`, 2.4-3.3 MB each). The overnight driver that sequenced the runs and switched the server
  between configs is not published either (it is specific to our hosts); it only called `run.sh` as shown in the
  `.cmd` files.

Helpers (Python 3, standard library only):

- `rr01_probe.py <RR-01 transcript.json> [N] <out.json>`: replays spark-bench's RR-01 history with `content: null` and
  with `content: "None"`, N seeds each at temperature 0.3, thinking off, and counts SIGTERM-only replies.
  `BASE_URL` defaults to `http://127.0.0.1:8000/v1`.
- `teb_summ.py <teb.json>...`: one line per tool-eval-bench run (score, points, deployability, responsiveness, median
  turn, categories) plus the multi-step scenarios. Needs the bench's full JSON (`duration_seconds`).
- `sb_summ.py <run dir>...`: TrueScore line, domains and per-scenario AG scores from `sb/spark_bench.csv` and the run's
  `sb.log` (the log is not published, so run it on your own runs).
