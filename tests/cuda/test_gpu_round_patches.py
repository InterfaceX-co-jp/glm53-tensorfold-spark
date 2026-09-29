"""patches/0450 (GLM53_TF_GPU_ROUND): GPU-resident decode rounds.

CPU, Triton's interpreter (TRITON_INTERPRET=1; ``tests/gpuround_interp.py``: a correctly rounded fma, launches
serialized across threads):

- the knob's parsing;
- the device glue against the host rules it replaces, on random inputs: ``_accept_kernel`` (``Stepper.accept``'s keep
  rule with drafts, end tokens and stop-at-EOS, ``Stepper.done``, counters, the next window's room), ``_stage_kernel``
  (window ids, tie routes, sampler positions), ``_mtp_next_kernel`` (``DepthOptimizer.mtp_next`` and ``decode.draft``'s
  confidence chain), ``_f_depth_kernel`` (``DepthOptimizer.f_depth`` and ``Drafter.chain``'s confidence cut),
  ``_conv_shift_dev`` (``forward._conv_shift``), ``_backlog_kernel`` (``Stepper.accept``'s backlog copies);
- (``test_gpu_round_resident.py``) the real ``Batcher`` with resident rounds on a fake model, one rank and two ranks in
  threads.

GPU (skipped here): ``kda.replay_slots`` == ``replay_layers`` per slot and ``chain_slots`` == replay + ``chain`` bit
for bit on random states, and the synthetic checkpoint's replies with the knob == serial.

Run: TRITON_INTERPRET=1 PYTHONPATH=<patched TensorFold>/src:tests:tests/cuda \
    pytest -q tests/cuda/test_gpu_round_patches.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from tensorfold.families.glm5_next.cuda import gpuround as gr  # noqa: E402

CUDA = torch.cuda.is_available()
INTERP = os.environ.get("TRITON_INTERPRET") == "1" and type(gr._accept_kernel).__name__ == "InterpretedFunction"
interp = pytest.mark.skipif(not INTERP or CUDA, reason="Triton's CPU interpreter, no GPU")
gpu = pytest.mark.skipif(not CUDA, reason="CUDA only")

COSTS = {"verify": [29.2, 38.9, 45.2, 55.2, 59.5, 63.2, 65.6, 69.2, 72.0, 75.1, 78.0, 80.5, 83.0, 85.2, 87.0, 89.1],
         "mtp": 1.7, "mtp_step": 1.5, "mtp_row": 0.1, "block": 3.0, "taps_row": 0.05}


@pytest.fixture(autouse=True, scope="module")
def _interp():
    if INTERP:
        import gpuround_interp

        gpuround_interp.install()


# -- the knob ----------------------------------------------------------------------------------------------------------
def test_parse():
    assert not gr.parse(None).on and not gr.parse("0").on
    a = gr.parse("1")
    assert a.sample and a.resident and a.kda and a.describe() == "sample,resident,kda" and a.code() == 7
    b = gr.parse("resident")
    assert b.sample and b.resident and not b.kda               # resident implies the device sampler
    assert gr.parse("sample").code() == 1
    with pytest.raises(ValueError, match="unknown part"):
        gr.parse("sample,fast")
    with pytest.raises(ValueError, match="add 'resident'"):
        gr.parse("kda")


# -- accept ------------------------------------------------------------------------------------------------------------
def _host_accept(pend, count, maxtok, stop, done, drafts, sampled, eos, bound, arm):
    """``Stepper.accept`` + ``Stepper.done`` + the next window's room (``max(1, min(depth, room))``)."""

    if done:
        keep = 0
    else:
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop and sampled[i] in eos):
                break
            keep += 1
    count2 = count + keep
    last = sampled[keep - 1] if keep else pend
    done2 = bool(done) or count2 >= maxtok or (bool(stop) and keep > 0 and last in eos)
    cap = 0 if done2 or arm == 0 else max(1, min(bound, maxtok - count2))
    return keep, count2, done2, last, cap


@interp
def test_accept_kernel_is_the_steppers_rule():
    rng = np.random.default_rng(1)
    n, backlog = 5, 32
    eos = [3, 7]
    for trial in range(40):
        I = torch.zeros((n, gr.NI), dtype=torch.int32)
        H = torch.zeros((n, gr.NH), dtype=torch.int32)
        F = torch.ones((n, gr.NF), dtype=torch.float64)
        DRAFT = torch.zeros((n, gr.MAXD), dtype=torch.int32)
        MTOK = torch.zeros((n, backlog + 1), dtype=torch.int32)
        active = sorted(rng.choice(n, size=int(rng.integers(1, n + 1)), replace=False).tolist())
        TOK = torch.zeros((80,), dtype=torch.int32)
        want, o = {}, 0
        for s in active:
            nd = int(rng.integers(0, 8))
            R = nd + 1 + int(rng.integers(0, 3))                       # padded rows past the drafts
            drafts = [int(x) for x in rng.integers(0, 10, size=nd)]
            sampled = [int(x) for x in rng.integers(0, 10, size=R)]
            for i in range(nd):                                        # mostly accepted prefixes
                if rng.random() < 0.7:
                    sampled[i] = drafts[i]
            pend, count, maxtok = int(rng.integers(0, 10)), int(rng.integers(1, 40)), int(rng.integers(2, 50))
            stop, done = int(rng.random() < 0.5), int(rng.random() < 0.15)
            mcount, fcount = int(rng.integers(0, 10)), int(rng.integers(0, 10))
            bound, arm = int(rng.integers(0, 8)), int(rng.integers(0, 3))
            I[s, gr.PEND], I[s, gr.COUNT], I[s, gr.MAXTOK], I[s, gr.STOPEOS] = pend, count, maxtok, stop
            I[s, gr.DONE], I[s, gr.NREAL], I[s, gr.MCOUNT], I[s, gr.FCOUNT] = done, nd, mcount, fcount
            I[s, gr.POS] = 1000 + s
            H[s, gr.RUN], H[s, gr.OFF], H[s, gr.BOUND], H[s, gr.ARMN] = R, o, bound, arm
            DRAFT[s, :nd] = torch.tensor(drafts, dtype=torch.int32)
            TOK[o:o + R] = torch.tensor(sampled, dtype=torch.int32)
            want[s] = (_host_accept(pend, count, maxtok, stop, done, drafts, sampled, eos, bound, arm), sampled, mcount,
                       fcount, R, nd)
            o += R
        slots = torch.tensor(active, dtype=torch.int32)
        KEEPV = torch.zeros((n,), dtype=torch.int32)
        rec_w = gr.REC_HEAD + 80 + gr.MAXD
        REC = torch.zeros((n + 1, rec_w), dtype=torch.int32)
        EOS = torch.tensor(eos, dtype=torch.int32)
        gr._accept_kernel[(len(active),)](I, H, F, DRAFT, TOK, EOS, len(eos), KEEPV, MTOK, REC, slots,
                                         BACKLOG_=backlog, NI_=gr.NI, NH_=gr.NH, NF_=gr.NF, MAXD_=gr.MAXD, REC_W=rec_w,
                                         MAXR=80)
        for i, s in enumerate(active):
            (keep, count2, done2, last, cap), sampled, mcount, fcount, R, nd = want[s]
            assert int(KEEPV[i]) == keep and int(I[s, gr.KEEP]) == keep
            assert int(I[s, gr.COUNT]) == count2 and bool(I[s, gr.DONE]) == done2 and int(I[s, gr.PEND]) == last
            assert int(I[s, gr.POS]) == 1000 + s + keep and int(I[s, gr.PREKEEP]) == keep
            assert int(I[s, gr.MOFF]) == mcount and int(I[s, gr.MCOUNT]) == min(mcount + keep, backlog)
            assert int(I[s, gr.FOFF]) == fcount and int(I[s, gr.FCOUNT]) == min(fcount + keep, backlog)
            assert MTOK[s, mcount:mcount + keep].tolist() == sampled[:keep]
            assert int(I[s, gr.CAP]) == cap and int(I[s, gr.LIVE]) == int(cap > 0) and int(I[s, gr.NNEXT]) == 0
            r = REC[i].tolist()
            assert r[:6] == [keep, count2, int(done2), 1000 + s + keep, R, nd]
            assert r[gr.REC_HEAD:gr.REC_HEAD + R] == sampled


@interp
def test_stage_kernel_windows_routes_positions():
    n = 3
    I = torch.zeros((n, gr.NI), dtype=torch.int32)
    H = torch.zeros((n, gr.NH), dtype=torch.int32)
    DRAFT = torch.zeros((n, gr.MAXD), dtype=torch.int32)
    spec = {0: (11, [5, 6], 4, 100), 2: (12, [], 3, 200), 1: (13, [9, 8, 7], 4, 300)}      # pending, drafts, rows, pos
    rows_slot, rows_idx, o = [], [], 0
    for s, (p, d, R, pos) in spec.items():
        I[s, gr.PEND], I[s, gr.NREAL], I[s, gr.POS] = p, len(d), pos
        DRAFT[s, :len(d)] = torch.tensor(d, dtype=torch.int32) if d else DRAFT[s, :0]
        H[s, gr.OFF] = o
        rows_slot += [s] * R
        rows_idx += list(range(R))
        o += R
    T = o
    IDS = torch.zeros((T,), dtype=torch.int32)
    SRC = torch.zeros((T,), dtype=torch.int64)
    INT = torch.zeros((T, 8), dtype=torch.int64)
    gr._stage_kernel[(T,)](I, H, DRAFT, torch.tensor(rows_slot, dtype=torch.int32),
                           torch.tensor(rows_idx, dtype=torch.int32), IDS, SRC, INT, NI_=gr.NI, NH_=gr.NH,
                           MAXD_=gr.MAXD)
    assert IDS.tolist() == [11, 5, 6, 6, 12, 12, 12, 13, 9, 8, 7]      # padded rows repeat the last real token
    assert SRC.tolist() == [0, 1, 2, 2, 4, 4, 4, 7, 8, 9, 10]          # and route as its row
    assert INT[:, 3].tolist() == [101, 102, 103, 104, 201, 202, 203, 301, 302, 303, 304]


# -- depths ------------------------------------------------------------------------------------------------------------
def _optimizer(rng):
    from tensorfold.families.glm5_next.cuda.depth import DepthOptimizer

    opt = DepthOptimizer(COSTS, most_m=15, most_f=15)
    for _ in range(int(rng.integers(0, 30))):                  # some calibration history
        arm = "m" if rng.random() < 0.5 else "f"
        k = int(rng.integers(1, 8))
        opt.used = [float(x) for x in rng.uniform(0.05, 1.0, size=k)]
        opt.record(arm, k + 1, 2, 3, int(rng.integers(1, k + 2)))
    return opt


def _depth_state(opt, conf: float, cap: int, n: int = 1):
    from tensorfold.families.glm5_next.cuda.resident import depth_params

    I = torch.zeros((n, gr.NI), dtype=torch.int32)
    F = torch.ones((n, gr.NF), dtype=torch.float64)
    P = np.zeros((n, gr.NP), dtype=np.float64)
    for s in range(n):
        depth_params(P[s], opt, conf)
        I[s, gr.CAP] = cap
        I[s, gr.LIVE] = int(cap > 0)
    return I, F, torch.from_numpy(P)


@interp
@pytest.mark.parametrize("mode", ["opt", "conf"])
def test_mtp_chain_decisions_are_the_hosts(mode):
    rng = np.random.default_rng(3 if mode == "opt" else 4)
    for trial in range(60):
        opt = _optimizer(rng) if mode == "opt" else None
        conf = 0.0 if mode == "opt" else float(rng.choice([0.0, 0.35, 0.6]))
        count = int(rng.integers(1, 9))
        probs = [float(x) for x in rng.uniform(0.02, 1.0, size=count)]
        # host: decode.draft's loops
        drafts, chained = 0, 0
        if opt is not None:
            opt.mtp_begin()
            for j in range(count):
                take, more = opt.mtp_next(j, probs[j], count)
                if not take:
                    break
                drafts += 1
                if not more:
                    break
                chained += 1
        else:
            chain = 1.0
            for j in range(count):
                p = probs[j]
                if conf > 0 and j > 0 and chain * p < conf:
                    break
                drafts += 1
                if conf > 0:
                    chain *= p
                    if chain < conf:
                        break
                if j + 1 < count:
                    chained += 1
                else:
                    break
        # device: every step the host launched (a stale bound: ``count`` of them), the kernel deciding
        I, F, P = _depth_state(opt, conf, count)
        NEXT = torch.zeros((1, gr.MAXD), dtype=torch.int32)
        NPROB = torch.zeros((1, gr.MAXD), dtype=torch.float64)
        slots = torch.zeros((1,), dtype=torch.int32)
        for j in range(count):
            tok = torch.tensor([100 + j], dtype=torch.int32)
            prob = torch.tensor([probs[j]], dtype=torch.float64)
            gr._mtp_next_kernel[(1,)](I, F, P, NEXT, NPROB, tok, prob, slots, j, NI_=gr.NI, NF_=gr.NF, NP_=gr.NP,
                                      MAXD_=gr.MAXD, QMAX=gr.Q_MAX)
        assert int(I[0, gr.NNEXT]) == drafts, (mode, probs, conf)
        assert NEXT[0, :drafts].tolist() == [100 + j for j in range(drafts)]
        assert NPROB[0, :drafts].tolist() == probs[:drafts]
        assert int(I[0, gr.CHAINED]) == chained, (mode, probs, conf)


@interp
@pytest.mark.parametrize("mode", ["opt", "conf"])
def test_dflash_depth_is_the_hosts(mode):
    rng = np.random.default_rng(5 if mode == "opt" else 6)
    for trial in range(60):
        opt = _optimizer(rng) if mode == "opt" else None
        conf = 0.0 if mode == "opt" else float(rng.choice([0.0, 0.3, 0.5]))
        n = int(rng.integers(1, 8))
        probs = [float(x) for x in rng.uniform(0.02, 1.0, size=n)]
        if opt is not None:
            want = opt.f_depth(probs)
        else:
            want, chain = 0, 1.0
            for d, p in enumerate(probs):                        # ``Drafter.chain``'s confidence cut
                if conf > 0:
                    chain *= p
                    if d > 0 and chain < conf:
                        break
                want += 1
        I, F, P = _depth_state(opt, conf, int(rng.integers(n, n + 3)))
        NPROB = torch.zeros((1, gr.MAXD), dtype=torch.float64)
        NPROB[0, :n] = torch.tensor(probs, dtype=torch.float64)
        I[0, gr.NNEXT] = n
        gr._f_depth_kernel[(1,)](I, P, NPROB, torch.zeros((1,), dtype=torch.int32), NI_=gr.NI, NP_=gr.NP,
                                 MAXD_=gr.MAXD, QMAX=gr.Q_MAX)
        assert int(I[0, gr.NNEXT]) == want, (mode, probs, conf)


# -- commits and backlogs ----------------------------------------------------------------------------------------------
@interp
def test_conv_shift_dev_is_the_hosts():
    from tensorfold.families.glm5_next.cuda import forward

    rng = np.random.default_rng(7)
    L, C, rows, n = 3, 40, 6, 2
    for keep in range(0, rows + 1):
        conv = torch.from_numpy(rng.standard_normal((L + 1, 3, C)).astype(np.float32)).to(torch.bfloat16)
        proj = torch.from_numpy(rng.standard_normal((L, rows, C + 7)).astype(np.float32)).to(torch.bfloat16)
        a, b = conv.clone(), conv.clone()
        I = torch.zeros((n, gr.NI), dtype=torch.int32)
        I[1, gr.KEEP] = keep
        gr._conv_shift_dev[(L, 1)](b, proj, I, 1, b.stride(0), proj.stride(0), proj.stride(1), C=C, TAPS=3,
                                   NI_=gr.NI, BLOCK=64)
        if keep:
            forward._conv_shift[(L, 1)](a, proj, keep, a.stride(0), proj.stride(0), proj.stride(1), C=C, TAPS=3,
                                        BLOCK=64)
        assert torch.equal(a, b), keep


@interp
def test_backlog_copies_the_kept_rows():
    n, D, backlog = 3, 20, 8
    rows = torch.arange(10 * D, dtype=torch.float32).view(10, D).to(torch.bfloat16)
    dst = torch.zeros((n, backlog + 1, 2 * D), dtype=torch.bfloat16)
    I = torch.zeros((n, gr.NI), dtype=torch.int32)
    H = torch.zeros((n, gr.NH), dtype=torch.int32)
    I[0, gr.KEEP], I[0, gr.MOFF], H[0, gr.OFF] = 2, 3, 0
    I[2, gr.KEEP], I[2, gr.MOFF], H[2, gr.OFF] = 3, 0, 5
    slots = torch.tensor([0, 2], dtype=torch.int32)
    gr._backlog_kernel[(2, 4, 1)](rows, rows.stride(0), dst, dst.stride(0), dst.stride(1), I, H, slots, D, D, backlog,
                                  NI_=gr.NI, NH_=gr.NH, WHICH=0, BLOCK=32)
    assert torch.equal(dst[0, 3:5, D:], rows[0:2]) and torch.equal(dst[2, 0:3, D:], rows[5:8])
    assert dst[0, :3].abs().sum() == 0 and dst[0, 5:].abs().sum() == 0 and dst[1].abs().sum() == 0
    assert dst[:, :, :D].abs().sum() == 0                     # the column offset: another tap's place


# -- GPU: the KDA kernels ----------------------------------------------------------------------------------------------
def _kda_case(seed: int, rows: int, H: int = 4, L: int = 2):
    """A random KDA layer set on the GPU: projections, gates, conv windows, states (the ``kda.chain`` inputs)."""

    from tensorfold.families.glm5_next.cuda import kda as kda_mod

    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    C = 3 * H * 128
    width = C + 2 * H * 128 + H + 64
    cs = (torch.randn((L, 3, C), generator=g, device=dev) * 0.5).to(torch.bfloat16)
    cw = (torch.randn((L, C, 4), generator=g, device=dev) * 0.5).to(torch.bfloat16)
    P = (torch.randn((rows, width), generator=g, device=dev)).to(torch.bfloat16)
    A = (torch.randn((rows, H * 128), generator=g, device=dev)).to(torch.bfloat16)
    Gt = (torch.randn((rows, H * 128), generator=g, device=dev)).to(torch.bfloat16)
    state = torch.randn((2, L, H, 128, 128), generator=g, device=dev) * 0.1
    a_log = torch.randn((L, H), generator=g, device=dev) * 0.1
    dt = torch.randn((L, H * 128), generator=g, device=dev) * 0.1
    norm = (1 + 0.1 * torch.randn((L, 128), generator=g, device=dev)).to(torch.bfloat16)
    ss = kda_mod.KDAScratchSet(L, rows, H, dev)
    return dict(cs=cs, cw=cw, P=P, A=A, G=Gt, state=state, a_log=a_log, dt=dt, norm=norm, ss=ss, C=C, H=H, L=L,
                b_off=C + 2 * H * 128)


@gpu
@pytest.mark.parametrize("rows", [1, 3, 8, 16])
def test_replay_slots_equals_replay_layers(rows):
    from tensorfold.families.glm5_next.cuda import kda as kda_mod

    cases = [_kda_case(10 + i, rows) for i in range(3)]
    for c in cases:
        for li in range(c["L"]):
            kda_mod.chain(c["P"], c["b_off"], c["A"], c["G"], c["cs"][li], c["cw"][li], c["state"][0, li],
                          c["a_log"][li], c["dt"][li], c["norm"][li], 1e-6, -5.0, rows, c["ss"].views[li],
                          c["state"][1, li])
    for keep in range(0, rows + 1):
        want = []
        for c in cases:
            out = torch.empty_like(c["state"][1])
            kda_mod.replay_layers(c["state"][0], c["ss"], keep, out)
            want.append(out)
        outs = [torch.empty_like(c["state"][1]) for c in cases]
        tab = torch.tensor([[c["state"][0].data_ptr(), o.data_ptr(), c["ss"].k.data_ptr(), c["ss"].v.data_ptr(),
                             c["ss"].g.data_ptr(), c["ss"].b.data_ptr(), c["ss"].k.stride(0), c["ss"].b.stride(0)]
                            for c, o in zip(cases, outs)], dtype=torch.int64, device="cuda")
        keeps = torch.full((3,), keep, dtype=torch.int32, device="cuda")
        run = torch.full((3,), rows + 1, dtype=torch.int32, device="cuda")
        kda_mod._ext().replay_slots(tab, keeps, run, cases[0]["L"], cases[0]["H"], cases[0]["state"][0, 0].numel(), 1)
        for o, wnt in zip(outs, want):
            assert torch.equal(o, wnt), keep


@gpu
@pytest.mark.parametrize("pre,rows", [(0, 3), (2, 3), (5, 8), (16, 1)])
def test_chain_slots_equals_replay_then_chain(pre, rows):
    """The folded chain (previous window's ``pre`` kept rows replayed in the prologue, then the window) gives the
    replayed state and the chain's outputs and saves bit for bit."""

    from tensorfold.families.glm5_next.cuda import kda as kda_mod

    prev = _kda_case(20, max(pre, 1))
    cur = _kda_case(21, rows)
    H, L = cur["H"], cur["L"]
    for li in range(L):                                   # the previous window's saves
        kda_mod.chain(prev["P"], prev["b_off"], prev["A"], prev["G"], prev["cs"][li], cur["cw"][li],
                      cur["state"][0, li], cur["a_log"][li], cur["dt"][li], cur["norm"][li], 1e-6, -5.0,
                      max(pre, 1), prev["ss"].views[li], prev["state"][1, li])
    # reference: replay, then the chain from the replayed state
    base = torch.empty_like(cur["state"][0])
    kda_mod.replay_layers(cur["state"][0], prev["ss"], pre, base)
    ref_ss = kda_mod.KDAScratchSet(L, rows, H, "cuda")
    ref_out = []
    for li in range(L):
        y = kda_mod.chain(cur["P"], cur["b_off"], cur["A"], cur["G"], cur["cs"][li], cur["cw"][li], base[li],
                          cur["a_log"][li], cur["dt"][li], cur["norm"][li], 1e-6, -5.0, rows, ref_ss.views[li],
                          torch.empty_like(base[li]))
        ref_out.append(y.clone())
    # folded
    ss = kda_mod.KDAScratchSet(L, rows, H, "cuda")
    state_out = torch.empty_like(cur["state"][0])
    out = torch.empty((L, rows, H * 128), dtype=torch.bfloat16, device="cuda")
    proj = torch.empty((L, rows, cur["P"].shape[1]), dtype=torch.bfloat16, device="cuda")
    I = torch.zeros((1, gr.NI), dtype=torch.int32, device="cuda")
    I[0, gr.PREKEEP] = pre
    I[0, gr.NREAL] = rows - 1
    for li in range(L):
        a, pa = ss.views[li], prev["ss"].views[li]
        tab = torch.tensor([[cur["P"].data_ptr(), cur["P"].stride(0), cur["A"].data_ptr(), cur["A"].stride(0),
                             cur["G"].data_ptr(), cur["G"].stride(0), cur["cs"][li].data_ptr(),
                             cur["state"][0, li].data_ptr(), state_out[li].data_ptr(), out[li].data_ptr(),
                             a.k.data_ptr(), a.v.data_ptr(), a.g.data_ptr(), a.b.data_ptr(), rows, pa.k.data_ptr(),
                             pa.v.data_ptr(), pa.g.data_ptr(), pa.b.data_ptr(), proj[li].data_ptr(), proj.stride(1),
                             0, 0, 0]], dtype=torch.int64, device="cuda")
        kda_mod._ext().chain_slots(tab, I, gr.NI, torch.zeros((1,), dtype=torch.int32, device="cuda"), gr.PREKEEP,
                                   gr.NREAL, H, cur["b_off"], cur["cw"][li], cur["a_log"][li], cur["dt"][li],
                                   cur["norm"][li], 1e-6, -5.0)
    assert torch.equal(state_out, base)
    for li in range(L):
        assert torch.equal(out[li], ref_out[li])
        for name in ("k", "v", "g", "b"):
            assert torch.equal(getattr(ss.views[li], name)[:rows], getattr(ref_ss.views[li], name)[:rows])
        assert torch.equal(proj[li, :, :cur["C"]], cur["P"][:, :cur["C"]])


# -- GPU: the synthetic checkpoint (one GPU playing rank 0 of two) -----------------------------------------------------
def _gpu_engine(path, parts: str, **kw):
    """``test_batch2_patches._engine`` with GLM53_TF_GPU_ROUND=``parts`` (and 0370's overlap, which resident needs)."""

    from test_batch2_patches import _engine as engine

    from tensorfold.families.glm5_next.cuda import decode_overlap as dover

    with pytest.MonkeyPatch.context() as m:
        m.setenv("GLM53_TF_GPU_ROUND", parts)
        m.setenv("GLM53_TF_DECODE_OVERLAP", "1" if parts not in ("", "0") else "0")
        m.setenv("GLM53_TF_BATCH_MTP", "1")
        m.setenv("GLM53_TF_BATCH_ROW_MS", "6.5")
        m.setenv("GLM53_TF_LOOKUP", "0")
        gr.env.cache_clear()
        dover.env.cache_clear()
        try:
            return engine(path, **kw)
        finally:
            gr.env.cache_clear()
            dover.env.cache_clear()


@pytest.fixture(scope="module")
def gckpt(tmp_path_factory):
    from test_glm_engine import _checkpoint, _drafter

    path = tmp_path_factory.mktemp("glm_gpuround")
    _checkpoint(path / "model", exl3=True)
    _drafter(path / "dflash2")
    return path


@pytest.fixture(scope="module")
def gref(gckpt):
    return _gpu_engine(gckpt, "0")


@gpu
def test_the_load_check_passes_on_this_gpu():
    from tensorfold.families.glm5_next.cuda import gpusample

    assert gpusample.self_check("cuda", rows=65536) is None


@gpu
@pytest.mark.parametrize("parts", ["sample", "resident", "1"])
@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
def test_four_requests_equal_serial(gckpt, gref, parts, greedy):
    """4 requests (auto with cost depths, of, om, a threshold policy) in shared rounds with the knob: every reply ==
    serial decoding; resident rounds ran (resident parts), drained when requests ended, and ran again."""

    from test_batch2_patches import _prompt, _sampling
    from test_batch_parallel_patches import _serial

    e = _gpu_engine(gckpt, parts, batch=4)
    try:
        assert e.w.meta.get("gpu_sample") is True
        sampling = _sampling(greedy)
        prompts = [_prompt(470 + i, 30 + 17 * i) for i in range(4)]
        want = [_serial(gref, p, sampling, 64) for p in prompts]
        for quad in (("o", "o", "of", "om"), ("o", "c3:0.35", "fc5:0.3", "0")):
            got = e.batch.generate_batch([dict(prompt=p, max_tokens=64, sampling=sampling, policy=pol)
                                          for p, pol in zip(prompts, quad)])
            assert [t for t, _ in got] == want, quad
        res = e.batch.resident
        if parts in ("resident", "1"):
            assert res is not None and res.rounds > 20 and res.entries >= 2, (res and res.drains)
            assert res.kda == (parts == "1")
        else:
            assert res is None
    finally:
        del e
        torch.cuda.empty_cache()


@gpu
def test_a_lone_request_and_a_session_resume(gckpt, gref):
    """One request alone (slot 0, the production single stream) with every part: == serial; then a second turn resumed
    from its reply snapshot (the KDA state materialized when resident rounds drained) == a fresh prefill of it."""

    from test_batch2_patches import _prompt, _run, _sampling

    e = _gpu_engine(gckpt, "1", batch=4)
    try:
        for greedy in (True, False):
            s = _sampling(greedy)
            p = _prompt(480 + int(greedy), 90)
            first, _ = _run(e, p, s, policy="o", tokens=80)
            assert first == _run(gref, p, s, policy="0", tokens=80)[0]
            p2 = p + first + _prompt(490, 12)
            assert _run(e, p2, s, policy="o", tokens=40)[0] == _run(gref, p2, s, policy="0", tokens=40)[0]
        assert e.batch.resident.rounds > 30
    finally:
        del e
        torch.cuda.empty_cache()
