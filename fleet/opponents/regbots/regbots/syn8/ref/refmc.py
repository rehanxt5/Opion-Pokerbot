"""mcbase.py — Monte-Carlo equity decision engine (Approach 2 core).

Real-time equity vs a modeled opponent range + pot-odds decisions + principled
pot-fraction sizing. `params` carries the tunables; the opponent-modeling bot
(bot_om.py) adjusts them dynamically, bot_mc.py uses neutral defaults.
"""
import numpy as np
import refpb as pb

# neutral parameters (bot_mc) — adjusted dynamically by the overlay bot
NEUTRAL = {
    "bluff_mult": 1.0,      # bluff frequency multiplier
    "call_mult": 1.0,       # required-equity multiplier (<1 = call looser)
    "size_mult": 1.0,       # bet sizing multiplier
    "value_thin": 0.0,      # extra thin-value threshold shift
    "rfi_pct": 0.52,        # preflop raise cutoff (percentile)
    "defend_pct": 0.38,     # preflop call cutoff facing a raise
    "threebet_pct": 0.86,   # preflop 3-bet cutoff
    "rng": None,
}

BASE_BLUFF = 0.16             # base bluff frequency when checked into, weak hand
BASE_BLUFF_VS_BET = 0.08      # base bluff-raise frequency facing a bet


def _rng(tr):
    if tr.__dict__.get("_rng") is None:
        tr._rng = np.random.default_rng()
    return tr._rng


def _preflop_decide(tr, req, P):
    """pct = fraction of classes WORSE than ours (0..1, higher = stronger).
    p > 1-x means 'top x% of hands'."""
    hole3 = tr.hole3
    pct = pb.PREFLOP.percentile_of(hole3)
    n_raises = sum(1 for h in tr.history if h["action"] in ("raise",))
    legal = req["legal_actions"]
    to_call = req["to_call"]
    min_r, max_r = req.get("min_raise", 4), req.get("max_raise", 200)

    def raise_to(amount):
        amt = max(min_r, min(int(amount), max_r))
        if "raise" not in legal or amt < min_r or amt > max_r:
            return None
        return {"type": "action", "action": "raise", "amount": amt}

    rng = _rng(tr)

    # unopened pot (button to act) / BB option after a limp
    if n_raises == 0:
        if to_call == 0:            # BB option: check or raise
            if pct > 1 - 0.40 and "raise" in legal:
                r = raise_to(6)
                if r:
                    return r
            return {"type": "action", "action": "check"}
        if to_call <= 1:            # button facing the big blind
            if pct > 1 - 0.58 and "raise" in legal:
                r = raise_to(6)
                if r:
                    return r
            if pct > 0.30 and "call" in legal:
                return {"type": "action", "action": "call"}   # limp the rest
            if "call" in legal and pct > 0.22 and rng.random() < 0.3:
                return {"type": "action", "action": "call"}   # occasional loose limp
            return {"type": "action", "action": "fold"}

    # facing exactly one raise
    if n_raises == 1:
        if pct > P["threebet_pct"] and "raise" in legal and max_r >= 16:
            r = raise_to(16)
            if r:
                return r
        if pct > 1 - P["defend_pct"] * P["call_mult"]:
            if "call" in legal:
                return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}

    # facing 3bet or jam
    need = 0.72 if req["max_raise"] < 80 else 0.90
    if pct > 1 - need * P["call_mult"]:
        if "call" in legal:
            return {"type": "action", "action": "call"}
        if "check" in legal:
            return {"type": "action", "action": "check"}
    return {"type": "action", "action": "fold"}


def _street_equity(tr, req, rollouts):
    if tr.street == "flop":
        return pb.equity_flop3(tr.hole3, tr.board, rollouts=rollouts)
    if tr.street == "turn":
        return pb.equity_turn2(tr.hole2, tr.board, rollouts=rollouts)
    if tr.street == "river":
        return pb.equity_river2(tr.hole2, tr.board)
    return 0.5


def _postflop_decide(tr, req, P, rollouts):
    legal = req["legal_actions"]
    eq = _street_equity(tr, req, rollouts)
    pot = req["pot"]
    to_call = req["to_call"]
    min_r, max_r = req.get("min_raise", 2), req.get("max_raise", 200)

    def raise_to(frac):
        target = max(req["my_bet"], req["opp_bet"]) + max(2, int(round(frac * (pot + to_call))))
        amt = max(min_r, min(int(target), max_r))
        if "raise" not in legal or amt < min_r or amt > max_r:
            return None
        return {"type": "action", "action": "raise", "amount": amt}

    rng = _rng(tr)
    facing_bet = to_call > 0

    if not facing_bet:
        # we may check or bet
        bluffs = BASE_BLUFF * P["bluff_mult"]
        if eq > 0.68 + P["value_thin"]:
            r = raise_to(0.75 * P["size_mult"])
            return r or {"type": "action", "action": "check"}
        if eq > 0.55 + P["value_thin"]:
            r = raise_to(0.55 * P["size_mult"])
            return r or {"type": "action", "action": "check"}
        if eq > 0.42 and rng.random() < 0.45 * P["size_mult"]:
            r = raise_to(0.5 * P["size_mult"])
            if r:
                return r
            return {"type": "action", "action": "check"}
        if rng.random() < bluffs:
            r = raise_to(0.6 * P["size_mult"])
            if r:
                return r
        return {"type": "action", "action": "check"}

    # facing a bet: pot odds
    required = to_call / max(1.0, pot + to_call)
    need = required * P["call_mult"]
    if eq > 0.82:
        r = raise_to(1.0 * P["size_mult"])          # raise for value
        if r:
            return r
        return {"type": "action", "action": "call"}
    if eq > max(need, 0.36):
        return {"type": "action", "action": "call"}
    if rng.random() < BASE_BLUFF_VS_BET * P["bluff_mult"]:
        r = raise_to(1.25 * P["size_mult"])
        if r:
            return r
    return {"type": "action", "action": "fold"}


def decide(tr, req, P=None):
    P = P or NEUTRAL
    if tr.street == "preflop":
        return _preflop_decide(tr, req, P)
    # adapt computation depth to the time budget
    budget = tr.budget_ms(cap_ms=110)
    rollouts = 300 if budget > 60 else 120
    if tr.street == "flop":
        rollouts = min(400, max(120, rollouts))
    return _postflop_decide(tr, req, P, rollouts)


def discard_potential(kept2, board3, rollouts=130, rng=None):
    """Equity of a kept 2-card holding vs uniform opp 2-card over runouts."""
    rng = rng or np.random.default_rng()
    known = list(kept2) + list(board3)
    deck = pb._deck_excluding(known)
    nxt = pb._sample_cards(deck, 4, rollouts, rng)       # opp2 + turn + river
    opp2, run = nxt[:, 0:2], nxt[:, 2:4]
    board5 = np.concatenate([np.tile(np.array(board3, dtype=np.int8), (rollouts, 1)), run], axis=1)
    mine = np.concatenate([np.tile(np.array(kept2, dtype=np.int8), (rollouts, 1)), board5], axis=1)
    theirs = np.concatenate([opp2, board5], axis=1)
    return float(pb.showdown_win(mine, theirs).mean())


def choose_discard(tr, req, balance=0.22):
    """Discard the card whose removal maximizes kept-hand potential, with slight
    balancing so we don't always leak the same pattern."""
    hole3, board3 = tr.hole3, tr.board
    rng = _rng(tr)
    pots = []
    for i in range(3):
        kept = [c for j, c in enumerate(hole3) if j != i]
        pots.append(discard_potential(kept, board3, rollouts=120, rng=rng))
    order = sorted(range(3), key=lambda i: -pots[i])
    pick = order[0]
    if len(order) > 1 and pots[order[1]] > pots[order[0]] - 0.04 \
            and rng.random() < balance:
        pick = order[1]
    return {"type": "discard", "card": pb.i2c(hole3[pick])}
