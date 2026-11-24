"""Parametric style opponents for opion dataset diversity.

One shared implementation, five entry-point scripts (station/rock/maniac/
randombot/mixer) because the engine launches bots as bare `python path`.
Protocol: JSON lines on stdin (protocol 1.0); replies on stdout.

Styles (per-hand mode for mixer):
  station  — never raises; calls any bet it can afford; folds only unpayable
  rock     — plays pairs/B+-high only; min-raises sometimes when checked to;
             folds otherwise
  maniac   — min-raises whenever legal, calls otherwise, never folds
  random   — uniform over legal actions, uniform raise amount
  mixer    — picks station/maniac/rock per hand
"""
import json
import random
import sys

RANKS = "23456789TJQKA"


class Style:
    def __init__(self, kind):
        self.kind = kind
        self.mode = kind
        self.rng = random.Random(4242 + sum(ord(c) for c in kind))
        self.hole = []

    def on_new_hand(self, m):
        self.hole = m.get("hole_cards", [])
        if self.kind == "mixer":
            self.mode = self.rng.choice(["station", "maniac", "rock"])

    def _hi(self):
        return max(RANKS.index(c[0]) for c in self.hole)

    def _pair(self):
        rs = [c[0] for c in self.hole]
        return len(set(rs)) < len(rs)

    def act(self, req):
        legal = req["legal_actions"]
        tc = req["to_call"]
        pot = max(int(req.get("pot", 4)), 1)
        mode = self.mode

        def raise_to(mult=None):
            lo, hi = int(req["min_raise"]), int(req["max_raise"])
            amt = lo if mult is None else int(lo + mult * (hi - lo))
            return {"type": "action", "action": "raise", "amount": max(lo, min(hi, amt))}

        if mode == "maniac":
            if "raise" in legal:
                return raise_to()
            return {"type": "action", "action": "call" if tc > 0 else "check"}

        if mode == "random":
            a = self.rng.choice(legal)
            if a == "raise":
                return raise_to(self.rng.random())
            return {"type": "action", "action": a}

        if mode == "rock":
            strong = self._pair() or self._hi() >= 11  # pair or K/A-high
            if tc > 0:
                if strong and tc <= int(0.4 * pot):
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}
            if strong and self.rng.random() < 0.5 and "raise" in legal:
                return raise_to()
            return {"type": "action", "action": "check"}

        # station (default): never raises, calls anything affordable
        if tc > 0:
            if tc <= int(0.5 * req["my_stack"]) + 1:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}
        return {"type": "action", "action": "check"}

    def discard(self, req):
        hole = req["hole_cards"]
        return {"type": "discard",
                "card": min(hole, key=lambda c: RANKS.index(c[0]))}


def main(kind):
    st = Style(kind)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        m = json.loads(line)
        t = m.get("type")
        if t == "new_hand":
            st.on_new_hand(m)
        elif t == "action_request":
            sys.stdout.write(json.dumps(st.act(m)) + "\n")
            sys.stdout.flush()
        elif t == "discard_request":
            sys.stdout.write(json.dumps(st.discard(m)) + "\n")
            sys.stdout.flush()
        elif t == "match_end":
            break


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "station")
