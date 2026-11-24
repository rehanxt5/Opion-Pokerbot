"""tns.py — G85_TNS transformer opponent-range prior (pure numpy inference).

A tiny self-attention transformer (trained offline by scripts/g85_tns_train.py,
weights in data/tns_w.npz) predicts the opponent's made-hand category
distribution (9 classes) at the current street from the hand's action-token
sequence + numeric context.  The posterior re-weights the Bayesian opponent
range sample by likelihood ratio, exactly mirroring the nnu (g82) scheme.

Feature semantics MUST mirror scripts/g85_tns_extract.py (train/serve parity):
  tokens: (actor, street, action, amount-bucket) events, max 48
  nums:   19 floats (street, our hole quality, board texture, pot, to_call,
          raises, seat, their exposed discard)
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_WPATH = os.path.join(_HERE, "data", "tns_w.npz")

D = 32
DFF = 64
NH = 4
DH = D // NH
MAXTOK = 48
NCLS = 9
V = 2 * 4 * 4 * 6 + 2
N_NUM = 19
POOL = 2 * D

_STREETS = ["preflop", "flop", "turn", "river"]
_ACTS = {"raise": 0, "call": 1, "check": 2, "fold": 3}   # extractor order r,c,x,f
RANKS = "23456789TJQKA"
_R2I = {r: i for i, r in enumerate(RANKS)}

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def _silu(z):
    return z * _sigmoid(z)

def _softmax_last(S, mask):
    S = np.where(mask, S, -1e9)
    S = S - S.max(-1, keepdims=True)
    A = np.exp(S)
    return A / (A.sum(-1, keepdims=True) + 1e-9)

def _amt_bucket(a):
    if not a:
        return 0
    a = int(a)
    if a <= 3: return 1
    if a <= 7: return 2
    if a <= 15: return 3
    if a <= 40: return 4
    return 5

def _ev_tok(actor, street, act, amt):
    a = 0 if actor == "me" else 1
    s = _STREETS.index(street)
    ac = _ACTS.get(act, 0)
    b = _amt_bucket(amt)
    return a * 96 + s * 24 + ac * 6 + b + 1

def _eval5(c0, c1, c2, c3, c4):
    """Category-only 5-card evaluator on card INTS (rank<<2|suit)."""
    cs = [c0, c1, c2, c3, c4]
    rs = sorted((c >> 2 for c in cs), reverse=True)
    flush = len(set(c & 3 for c in cs)) == 1
    uniq = sorted(set(rs), reverse=True)
    straight_high = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_high = 3
    if flush and straight_high is not None:
        return 8
    cnt = {}
    for r in rs:
        cnt[r] = cnt.get(r, 0) + 1
    g = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    if g[0][1] == 4:
        return 7
    if g[0][1] == 3 and g[1][1] == 2:
        return 6
    if flush:
        return 5
    if straight_high is not None:
        return 4
    if g[0][1] == 3:
        return 3
    if g[0][1] == 2 and g[1][1] == 2:
        return 2
    if g[0][1] == 2:
        return 1
    return 0

def _best2_of3(hole3, board3):
    """Our flop 'mine' proxy: best hole pair by current-board 5-card eval."""
    if not board3:
        return hole3[:2]
    best = (-1, (0, 1))
    for i in range(3):
        for j in range(i + 1, 3):
            sc = _eval5(hole3[i], hole3[j], board3[0], board3[1], board3[2])
            if (sc, (i, j)) > best:
                best = (sc, (i, j))
    return [hole3[best[1][0]], hole3[best[1][1]]]

def snapshot(street, board, history, my_discard, opp_discard,
             i_am_button, hole3):
    """Build (tokens, nums) at OUR decision point. board: card STRINGS."""
    toks = []
    pot = 3
    n_r_prior = 0
    n_r_street = 0
    commit = {"me": 0, "opp": 0}
    to_call = 0
    prev = None
    for h in history:
        st = h.get("street", "preflop")
        a = h.get("actor")
        ac = h.get("action")
        am = h.get("amount")
        if st != prev:                    # true street rollover
            if prev is not None:
                n_r_prior += n_r_street
                n_r_street = 0
                commit = {"me": 0, "opp": 0}
                to_call = 0
            prev = st
        toks.append(_ev_tok(a, st, ac, am))
        if ac in ("raise", "call") and am:
            inc = max(0, int(am) - commit[a])
            pot += inc
            commit[a] = int(am) if ac == "raise" else max(commit[a], int(am))
        if ac == "raise":
            n_r_street += 1
        if st == street:
            to_call = max(0, commit["opp"] - commit["me"])
    if prev != street:                # we act first on a fresh street
        n_r_prior += n_r_street
        n_r_street = 0
        commit = {"me": 0, "opp": 0}
        to_call = 0
    # our cards at this street
    if street == "flop" or my_discard is None:
        mine = _best2_of3(list(hole3), board[:3])
    else:
        mine = [c for c in hole3 if c != my_discard]
    # numeric features (mirror the extractor exactly)
    hi = max(c >> 2 for c in mine) / 12.0
    pair = 1.0 if (mine[0] >> 2) == (mine[1] >> 2) else 0.0
    suited = 1.0 if (mine[0] & 3) == (mine[1] & 3) else 0.0
    gap = abs((mine[0] >> 2) - (mine[1] >> 2)) / 12.0
    br = [c >> 2 for c in board]
    b_hi = (max(br) / 12.0) if br else 0.0
    rc = {}
    for r in br:
        rc[r] = rc.get(r, 0) + 1
    b_paired = 1.0 if any(v >= 2 for v in rc.values()) else 0.0
    bs = {}
    for c in board:
        bs[c & 3] = bs.get(c & 3, 0) + 1
    b_flush3 = 1.0 if (bs and max(bs.values()) >= 3) else 0.0
    m = 0
    for r in set(br):
        m |= 1 << r
    b_str3 = 0.0
    for sh in range(4, 13):
        if bin((m >> (sh - 4)) & 0b11111).count("1") >= 3:
            b_str3 = 1.0
            break
    d_hi = -0.5
    d_board = 0.0
    if street != "flop" and opp_discard is not None:
        d_hi = (opp_discard >> 2) / 12.0
        if board:
            d_board = 1.0 if any((opp_discard >> 2) == (c >> 2) for c in board) else 0.0
    nums = [
        {"flop": 1, "turn": 0, "river": 0}.get(street, 0),
        {"flop": 0, "turn": 1, "river": 0}.get(street, 0),
        {"flop": 0, "turn": 0, "river": 1}.get(street, 0),
        hi, pair, suited, gap,
        b_hi, b_paired, b_flush3, b_str3,
        min(pot, 200) / 200.0,
        min(to_call, 100) / 100.0,
        min(n_r_street, 3) / 3.0,
        min(n_r_prior, 4) / 4.0,
        1.0 if i_am_button else 0.0,
        min(len(board), 5) / 5.0,
        d_hi, d_board,
    ]
    if len(toks) > MAXTOK:
        toks = toks[:8] + toks[-(MAXTOK - 8):]
    return toks, nums

class TnsNet:
    def __init__(self, w):
        self.w = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}

    def forward(self, toks, nums):
        """Single-sample forward. Returns 9-class softmax posterior."""
        w = self.w
        toks = toks[-MAXTOK:]
        T = max(1, len(toks))
        tarr = np.zeros((T,), dtype=np.int64)
        tarr[:len(toks)] = toks
        mask = np.zeros((T,), dtype=bool)
        mask[:len(toks)] = True
        X = np.reshape(w["ev"][tarr] + w["pos"][:T][None], (T, D))
        Q = X @ w["Wq"]; K = X @ w["Wk"]; Vh = X @ w["Wv"]
        Qh = Q.reshape(T, NH, DH).transpose(1, 0, 2)     # (NH,T,DH)
        Kh = K.reshape(T, NH, DH).transpose(1, 0, 2)
        Vg = Vh.reshape(T, NH, DH).transpose(1, 0, 2)
        S = np.matmul(Qh, Kh.transpose(0, 2, 1)) / (DH ** 0.5)
        S = np.where(mask[None, None, :], S, -1e9)
        S = S - S.max(-1, keepdims=True)
        A = np.exp(S); A = A / (A.sum(-1, keepdims=True) + 1e-9)
        Oh = np.matmul(A, Vg)                            # (NH,T,DH)
        O = Oh.transpose(1, 0, 2).reshape(T, D) @ w["Wo"]
        H = X + O
        Z1 = H @ w["f1w"] + w["f1b"]
        F = _silu(Z1) @ w["f2w"] + w["f2b"]
        Y = H + F
        sc = (Y @ w["poolq"]) / (D ** 0.5)
        ap = _softmax_last(sc, mask)
        pooled = (ap[:, None] * Y).sum(0)
        cnt = max(mask.sum(), 1)
        meanp = (mask[:, None].astype(np.float64) * Y).sum(0) / cnt
        nums = np.asarray(nums, dtype=np.float64)
        feat = np.concatenate([pooled, meanp, nums])
        Zh1 = feat @ w["h1w"] + w["h1b"]
        h1 = _silu(Zh1)
        logits = h1 @ w["h2w"] + w["h2b"]
        z = logits - logits.max()
        e = np.exp(z)
        return e / e.sum()

# ---- likelihood-ratio weighting (nnu-style, conservative caps) ----
KL_MIN = 0.08
LAM_BASE = 0.22
LAM_GAIN = 0.13
LAM_MAX = 0.35
# g85v: amplified lam for confirmed-folder matchups (set via
# SYN_TNS_LAM env by synergy when the g85t station gate is open and the
# model's KL is live); base values remain the conservative default
LAM_MAX_AMP = float(os.environ.get("SYN_TNS_LAM_MAX_AMP", "0.35"))
W_CLIP = (0.25, 4.0)
KL_MARK = 1.2

def cat_weights(net, toks, nums, cats, n_show):
    """Per-sample likelihood-ratio weights over the opponent sample.

    cats: (N,) int array — each sample's current-street category.
    Returns (w, kl) or (None, 0.0).
    """
    N = cats.shape[0]
    if N < 60:
        return None, 0.0
    try:
        p = net.forward(toks, nums)
        q = np.bincount(cats, minlength=9).astype(np.float64)
        q = (q + 0.5) / (N + 4.5)
        q = q / q.sum()
        kl = float(np.sum(p * np.log((p + 1e-9) / (q + 1e-9))))
        if kl < KL_MIN:
            return None, kl
        lam = min(LAM_MAX_AMP, LAM_BASE + LAM_GAIN * min(1.0, np.log2(1.0 + n_show) / 8.0))
        ratio = (p + 1e-4) / (q + 1e-4)
        w = np.power(ratio[cats], lam)
        np.clip(w, W_CLIP[0], W_CLIP[1], out=w)
        return w, kl
    except Exception:
        return None, 0.0

_NET = None

def load():
    global _NET
    try:
        _NET = TnsNet(dict(np.load(_WPATH)))
        return _NET
    except Exception:
        _NET = None
        return None

def net():
    return _NET
