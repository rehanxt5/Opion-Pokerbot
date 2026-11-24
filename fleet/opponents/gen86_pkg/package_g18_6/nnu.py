"""nnu.py — Gen 8.2 neural range prior (NNU): small pure-numpy MLP that maps
the opponent's observed action line + board texture + measured profile to a
distribution over their made-hand CATEGORY (0 high card .. 8 straight flush).

Design goals
  - SELF-CONTAINED: stdlib + numpy only; imports nothing from pbcore/synergy
    so the SAME code path serves offline training-data extraction (log
    replay) and online play (feature parity by construction).
  - CATEGORY-LEVEL conditioning: the output re-weights the existing Bayesian
    opponent-range sample by a likelihood ratio p[cat]/q[cat] (q = the
    sample's own marginal category distribution).  It adds information; it
    never replaces the sample, the preflop bands, the discard-explain model,
    or the street-aggression filter.
  - GATED: env flag SYN_NNU (default on for 8.2); dormant while the g81
    honest-value hard gate is open or the g17 white-box exploit layer drives
    decisions; realized-EV governor closes it for the leg if materially
    active hands lose chips.

Card encoding (matches pbcore): int = rank_idx*4 + suit_idx, ranks 2..A ->
0..12, suits s h d c -> 0..3.

Feature vector (51 dims) — see snapshot(); every feature is computable both
from the engine's per-hand action history and from a replayed match log.
"""
import os
import numpy as np

OK = True                       # flipped False if weights fail to load
_NFEAT = 51
_CATS = 9
_STREETS = ("flop", "turn", "river")

_RANKS = "23456789TJQKA"
_SUITS = "shdc"
RANK_IDX = {r: i for i, r in enumerate(_RANKS)}
SUIT_IDX = {s: i for i, s in enumerate(_SUITS)}


def c2i(s):
    return RANK_IDX[s[0]] * 4 + SUIT_IDX[s[1]]


# ----------------------------------------------------------------- categories
_STR_BMS = []
for _hi in range(4, 13):         # 6-high (ranks 0-4) .. A-high (ranks 8-12)
    _bm = 0
    for _r in range(_hi - 4, _hi + 1):
        _bm |= 1 << _r
    _STR_BMS.append(_bm)
_STR_BMS.append(1 << 12 | 1 | 2 | 4 | 8)      # wheel A-2-3-4-5


def cat_best5(cards):
    """Best-5-of-N category (N in 5..7). 0 high card .. 8 straight flush.
    Semantics identical to pbcore eval7 categories / synergy._cat6_batch."""
    rc = [0] * 13
    sc = [0] * 4
    for c in cards:
        rc[c >> 2] += 1
        sc[c & 3] += 1
    has_quad = 4 in rc
    n_trips = sum(1 for x in rc if x == 3)
    n_pair = sum(1 for x in rc if x == 2)
    bm = 0
    for r in range(13):
        if rc[r]:
            bm |= 1 << r
    straight = any((bm & m) == m for m in _STR_BMS)
    flush_s = sc.index(max(sc)) if max(sc) >= 5 else -1
    flush = flush_s >= 0
    if flush:
        bmf = 0
        for c in cards:
            if (c & 3) == flush_s:
                bmf |= 1 << (c >> 2)
        sf = any((bmf & m) == m for m in _STR_BMS)
        if sf:
            return 8
    if has_quad:
        return 7
    if n_trips >= 1 and (n_pair >= 1 or n_trips >= 2):
        return 6
    if flush:
        return 5
    if straight:
        return 4
    if n_trips == 1:
        return 3
    if n_pair >= 2:
        return 2
    if n_pair == 1:
        return 1
    return 0


# ------------------------------------------------------------------- profile
class NNUProfile:
    """Cross-hand measured profile of the modeled opponent.

    Exactly-counted statistics with Laplace smoothing at read time; updated
    once per completed hand from PUBLIC information only (actions, exposed
    discards, showdowns).  The offline extractor updates it identically.
    """

    __slots__ = ("pf_raises", "pf_opts", "bet_n", "opt_n", "size_sum",
                 "bet_sd", "bet_sd_value", "cf_n", "cf_fold", "cr",
                 "n_show")

    def __init__(self):
        self.pf_raises = 0        # their preflop raises
        self.pf_opts = 0          # their preflop decision opportunities
        self.bet_n = 0            # their postflop bets+raises
        self.opt_n = 0            # their postflop action opportunities
        self.size_sum = 0.0       # sum of bet size / pot-before
        self.bet_sd = 0           # showdowns where they bet/raised postflop
        self.bet_sd_value = 0     # of those, shown cat >= 1 (pair+)
        self.cf_n = 0             # they checked, we bet: their responses
        self.cf_fold = 0          # of those, folds
        self.cr = 0               # their check-raises
        self.n_show = 0           # showdowns observed

    # ---- smoothed reads (feature-facing)
    def r_pf_raise(self):
        return (self.pf_raises + 1.5) / (self.pf_opts + 6.0)

    def r_bet_freq(self):
        return (self.bet_n + 2.0) / (self.opt_n + 8.0)

    def r_size(self):
        return (self.size_sum + 4.0) / (self.bet_n + 8.0)

    def r_value(self):
        return (self.bet_sd_value + 2.0) / (self.bet_sd + 4.0)

    def r_cf(self):
        return (self.cf_fold + 1.0) / (self.cf_n + 2.0)

    def r_cr(self):
        return (self.cr + 0.5) / (self.cf_n + 2.5)

    def r_show_log(self):
        import math
        return min(1.0, math.log2(1.0 + self.n_show) / 8.0)

    def r_sd_aggr(self):
        return (self.bet_sd + 1.0) / (self.n_show + 4.0)


def summarize(streets, showdown, opp_cards, opp_cat=None, button=None):
    """Normalize a completed hand into a profile-update event.

    `streets`: dict street -> list of {actor:"me"/"opp", action, amount?}
    (raw engine action history, per street, in order).  `showdown`: bool.
    `opp_cards`: the modeled player's shown 2 kept cards at showdown
    (ints) or None.  `opp_cat`: their final shown category (int or None).
    `button`: "me"/"opp" — who had the button this hand (blind sizes).
    Shared by the online hook and the offline log replay.
    """
    return {"streets": streets, "showdown": bool(showdown),
            "opp_cards": list(opp_cards) if opp_cards else None,
            "opp_cat": opp_cat, "button": button}


def observe_hand(P, ev):
    """Update profile P from a completed-hand event (see summarize).

    Single incremental walk over all streets tracking per-street commits
    and the running pot, exactly like the engine's street accounting.
    """
    streets = ev.get("streets") or {}
    button = ev.get("button") or "me"
    # ---- preflop
    pf = streets.get("preflop") or []
    if pf and any(a.get("actor") == "opp" for a in pf):
        P.pf_opts += 1
        if any(a.get("actor") == "opp" and a.get("action") == "raise"
               for a in pf):
            P.pf_raises += 1
    # ---- walk every street once, tracking commits and the running pot
    opp_post_bet = False
    pot = 0.0                                  # blinds enter via preflop sc
    for st in ("preflop", "flop", "turn", "river"):
        acts = streets.get(st) or []
        if not acts:
            continue
        if st == "preflop":
            sc = {"me": 1.0, "opp": 2.0} if button == "me" \
                else {"me": 2.0, "opp": 1.0}
        else:
            sc = {"me": 0.0, "opp": 0.0}
        if st != "preflop" and any(a.get("actor") == "opp" for a in acts):
            P.opt_n += 1
        for a in acts:
            act = a.get("action")
            who = a.get("actor", "me")
            if act == "raise":
                size = a.get("amount", 0) - sc[who]
                if who == "opp" and st != "preflop":
                    P.bet_n += 1
                    P.size_sum += size / max(1.0, pot)
                    opp_post_bet = True
                sc[who] = float(a.get("amount", 0))
            elif act == "call":
                sc[who] = sc["me" if who == "opp" else "opp"]
        # check-then-faced-our-bet outcomes (same street bookkeeping as the
        # g81 honest detector)
        if st != "preflop":
            opp_acts = [a for a in acts if a.get("actor") == "opp"]
            if opp_acts and opp_acts[0].get("action") == "check":
                first_ours = next((i for i, a in enumerate(acts)
                                   if a.get("actor") == "me"
                                   and a.get("action") == "raise"), None)
                if first_ours is not None:
                    r2 = next((a for a in acts[first_ours + 1:]
                               if a.get("actor") == "opp"), None)
                    if r2 is not None:
                        if r2.get("action") == "raise":
                            P.cr += 1
                            P.cf_n += 1
                        elif r2.get("action") in ("fold", "call"):
                            P.cf_n += 1
                            if r2.get("action") == "fold":
                                P.cf_fold += 1
        # close the street: both commits go into the pot
        pot += sc["me"] + sc["opp"]
    # ---- showdowns
    if ev.get("showdown") and ev.get("opp_cards") \
            and len(ev["opp_cards"]) == 2:
        P.n_show += 1
        if opp_post_bet:
            P.bet_sd += 1
            if ev.get("opp_cat") is not None and ev["opp_cat"] >= 1:
                P.bet_sd_value += 1
    return None


# ------------------------------------------------------------------ features
def _bucket_size(frac):
    """One-hot index 0..6 for their last bet size / pot-before."""
    if frac is None:
        return 0
    if frac <= 0.33:
        return 1
    if frac <= 0.55:
        return 2
    if frac <= 0.80:
        return 3
    if frac <= 1.20:
        return 4
    if frac <= 2.00:
        return 5
    return 6


def _first_action_class(street_acts, who):
    """Class of their first action on a street: 0 unreached, 1 check,
    2 call, 3 bet (first aggressive), 4 raise (over our bet)."""
    opp_seen_bet = False
    for a in street_acts:
        if a.get("actor") == who:
            act = a.get("action")
            if act == "check":
                return 1
            if act == "call":
                return 2
            if act == "raise":
                return 4 if opp_seen_bet else 3
            if act == "fold":
                return 0
        else:
            if a.get("action") == "raise":
                opp_seen_bet = True
    return 0


def snapshot(street, board, history, my_discard, opp_discard,
             i_am_button, P):
    """Feature vector at OUR decision point, modeling THEM.

    street: current street name (flop/turn/river)
    board:  visible board card ints (len 3/4/5)
    history: full per-hand action list so far, items
             {street, actor:"me"/"opp", action:"raise"/"call"/"check"/"fold",
              amount (total street commit for raises)}
    my_discard / opp_discard: our / their exposed discard card int or None
    i_am_button: bool (we posted the small blind 1)
    P: NNUProfile (match-level stats; frozen during the hand)
    """
    x = np.zeros(_NFEAT, dtype=np.float32)
    i = 0
    # street one-hot
    x[i + _STREETS.index(street)] = 1.0
    i += 3
    # ---- their line on the CURRENT street (actions before our decision)
    cur = [a for a in history if a.get("street") == street]
    first_cls = _first_action_class(cur, "opp")
    x[i + first_cls] = 1.0
    i += 5
    n_aggr = sum(1 for a in cur if a.get("actor") == "opp"
                 and a.get("action") == "raise")
    x[i + min(n_aggr, 2)] = 1.0
    i += 3
    # their last bet size / pot-before
    last_bet = None
    our_bet_this = False
    commits = {"me": 0.0, "opp": 0.0}
    # pot before this street (blinds enter via the preflop street commits)
    pot = 0.0
    for st in ("preflop", "flop", "turn", "river"):
        if st == street:
            break
        if st == "preflop":
            sc = {"me": 1.0, "opp": 2.0} if i_am_button \
                else {"me": 2.0, "opp": 1.0}
        else:
            sc = {"me": 0.0, "opp": 0.0}
        for a in history:
            if a.get("street") != st:
                continue
            if a.get("action") == "raise":
                sc[a.get("actor", "me")] = float(a.get("amount", 0))
            elif a.get("action") == "call":
                other = "opp" if a.get("actor") == "me" else "me"
                sc[a.get("actor", "me")] = sc[other]
        pot += sc["me"] + sc["opp"]
    # walk current street
    for a in history:
        if a.get("street") != street:
            continue
        who = a.get("actor", "me")
        act = a.get("action")
        if act == "raise":
            size = a.get("amount", 0) - commits[who]
            if who == "opp":
                last_bet = (size, pot + commits["me"] + commits["opp"])
            if who == "me":
                our_bet_this = True
            commits[who] = float(a.get("amount", 0))
        elif act == "call":
            commits[who] = commits["me" if who == "opp" else "opp"]
    frac = None
    if last_bet is not None:
        frac = last_bet[0] / max(1.0, last_bet[1])
    x[i + _bucket_size(frac)] = 1.0
    i += 7
    x[i] = 1.0 if (n_aggr >= 1 and our_bet_this) else 0.0     # raised our bet
    i += 1
    # ---- prior postflop streets (flop slot, turn slot)
    for slot in ("flop", "turn"):
        if slot == street:
            cls = 0
        else:
            acts = [a for a in history if a.get("street") == slot]
            reached = slot_before(slot, street) and len(acts) > 0
            cls = _first_action_class(acts, "opp") if reached else 0
        x[i + cls] = 1.0
        i += 5
    # ---- preflop
    pf = [a for a in history if a.get("street") == "preflop"]
    their_pf_raises = sum(1 for a in pf if a.get("actor") == "opp"
                          and a.get("action") == "raise")
    x[i + min(their_pf_raises, 2)] = 1.0
    i += 3
    their_last_pf_raise = None
    pf_commits = {"me": 1.0 if i_am_button else 2.0,
                  "opp": 2.0 if i_am_button else 1.0}
    for a in pf:
        who = a.get("actor", "me")
        if a.get("action") == "raise":
            if who == "opp":
                their_last_pf_raise = (a.get("amount", 0) - pf_commits["opp"],
                                       3.0)
            pf_commits[who] = float(a.get("amount", 0))
        elif a.get("action") == "call":
            pf_commits[who] = pf_commits["me" if who == "opp" else "me"]
    pf_cls = 0
    if their_last_pf_raise is not None:
        pf_cls = 1 if their_last_pf_raise[0] / 3.0 <= 0.30 else 2
    x[i + pf_cls] = 1.0
    i += 3
    # ---- board texture (visible cards)
    ranks = [c >> 2 for c in board]
    suits = [c & 3 for c in board]
    rc = [0] * 13
    for r in ranks:
        rc[r] += 1
    x[i] = 1.0 if max(rc) >= 2 else 0.0                      # paired
    i += 1
    suit_n = [0] * 4
    for s in suits:
        suit_n[s] += 1
    x[i] = 1.0 if max(suit_n) >= 3 else 0.0                  # 3+ suited
    i += 1
    x[i] = 1.0 if max(suit_n) >= 4 else 0.0                  # 4+ suited
    i += 1
    # connectedness: max #ranks in any 5-wide window / 5
    best = 0
    for hi in range(13):
        w = rc[max(0, hi - 4):hi + 1]
        best = max(best, sum(1 for v in w if v > 0))
    x[i] = best / 5.0
    i += 1
    x[i] = (max(ranks) / 12.0) if ranks else 0.0
    i += 1
    # ---- exposed discards (turn/river only)
    if street in ("turn", "river") and opp_discard is not None:
        x[i] = (opp_discard >> 2) / 12.0
        x[i + 1] = 1.0 if (opp_discard >> 2) >= 8 else 0.0
    else:
        x[i] = 0.5
    i += 2
    if street in ("turn", "river") and my_discard is not None:
        x[i] = (my_discard >> 2) / 12.0
    else:
        x[i] = 0.5
    i += 1
    # ---- profile (smoothed reads)
    x[i] = P.r_bet_freq(); i += 1
    x[i] = P.r_size(); i += 1
    x[i] = P.r_value(); i += 1
    x[i] = P.r_cf(); i += 1
    x[i] = P.r_cr(); i += 1
    x[i] = P.r_show_log(); i += 1
    x[i] = P.r_pf_raise(); i += 1
    x[i] = P.r_sd_aggr(); i += 1
    assert i == _NFEAT, (i, _NFEAT)
    return x


def slot_before(slot, street):
    return _STREETS.index(slot) < _STREETS.index(street)


# ------------------------------------------------------------------ the net
class NNNet:
    """3-layer MLP (51 -> 96 -> 64 -> 9), tanh hidden, softmax output.
    Feature standardization constants are stored with the weights."""

    def __init__(self, path):
        d = np.load(path)
        self.W1 = d["W1"].astype(np.float32)
        self.b1 = d["b1"].astype(np.float32)
        self.W2 = d["W2"].astype(np.float32)
        self.b2 = d["b2"].astype(np.float32)
        self.W3 = d["W3"].astype(np.float32)
        self.b3 = d["b3"].astype(np.float32)
        self.mu = d["mu"].astype(np.float32)
        self.sd = d["sd"].astype(np.float32)
        self.meta = {k: str(v) for k, v in d.get("meta", {}).items()} \
            if "meta" in d else {}

    def forward(self, x):
        z = (x - self.mu) / self.sd
        h1 = np.tanh(self.W1 @ z + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        o = self.W3 @ h2 + self.b3
        o -= o.max()
        e = np.exp(o)
        return e / e.sum()

    def forward_batch(self, X):
        Z = (X - self.mu) / self.sd
        H1 = np.tanh(Z @ self.W1.T + self.b1)
        H2 = np.tanh(H1 @ self.W2.T + self.b2)
        O = H2 @ self.W3.T + self.b3
        O -= O.max(axis=1, keepdims=True)
        E = np.exp(O)
        return E / E.sum(axis=1, keepdims=True)


_NN = None
_NN_PATH = None


def load(path=None):
    """Load the net once; returns NNNet or None (module degrades to no-op)."""
    global _NN, _NN_PATH, OK
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "nnu_w.npz")
    if _NN is not None and _NN_PATH == path:
        return _NN
    try:
        _NN = NNNet(path)
        _NN_PATH = path
        return _NN
    except Exception:
        OK = False
        return None


# ----------------------------------------------------- range conditioning
LAM_BASE = 0.35
LAM_GAIN = 0.55
LAM_MAX = 0.90
KL_MIN = 0.02                # below this the prior adds nothing
KL_MARK = 0.10               # "materially active" for the EV governor
W_CLIP = (0.25, 4.0)


def cat_weights(net, x, cats, n_show):
    """Per-sample likelihood-ratio weights over the opponent sample.

    cats: (N,) int array — each sample's current-street category.
    Returns (w, kl) or (None, 0.0).
    """
    N = cats.shape[0]
    if N < 150:
        return None, 0.0
    try:
        p = net.forward(x)                              # (9,)
        q = np.bincount(cats, minlength=9).astype(np.float64)
        q = (q + 0.5) / (N + 4.5)                       # smoothed marginal
        q = q / q.sum()
        kl = float(np.sum(p * np.log((p + 1e-9) / (q + 1e-9))))
        if kl < KL_MIN:
            return None, kl
        lam = min(LAM_MAX, LAM_BASE + LAM_GAIN *
                  min(1.0, np.log2(1.0 + n_show) / 8.0))
        ratio = (p + 1e-4) / (q + 1e-4)
        w = np.power(ratio[cats], lam)
        np.clip(w, W_CLIP[0], W_CLIP[1], out=w)
        return w, kl
    except Exception:
        return None, 0.0
