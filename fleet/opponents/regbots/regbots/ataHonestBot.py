#!/usr/bin/env python3
"""
Honest ABC Opponent:
Nature: Disciplined fundamental poker. Plays reasonable starting hands,
evaluates made hands and draws, bets for value, and folds when pot odds are unfavorable.
Card Intelligence: Uses full intelligent discard engine.
"""
import sys
import json
import random
from collections import Counter
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
FULL_DECK = [r + s for r in RANKS for s in SUITS]

def rank5(cards):
    values = sorted((RANK_VALUE[c[0]] for c in cards), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    uniq = sorted(set(values), reverse=True)
    straight_high = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight_high = 5
    groups = sorted(Counter(values).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [n for _, n in groups]
    by_group = tuple(v for v, _ in groups)

    if straight_high and flush:
        return (8, straight_high)
    if shape == [4, 1]:
        return (7,) + by_group
    if shape == [3, 2]:
        return (6,) + by_group
    if flush:
        return (5,) + tuple(values)
    if straight_high:
        return (4, straight_high)
    if shape == [3, 1, 1]:
        return (3,) + by_group
    if shape == [2, 2, 1]:
        return (2,) + by_group
    if shape == [2, 1, 1, 1]:
        return (1,) + by_group
    return (0,) + tuple(values)

def best_hand(hole, board):
    all_cards = list(hole) + list(board)
    if len(all_cards) < 5:
        return (0,)
    return max(rank5(combo) for combo in combinations(all_cards, 5))

def draw_bonus(hole2, board):
    all_cards = list(hole2) + list(board)
    suits = [c[1] for c in all_cards]
    suit_counts = Counter(suits)
    max_suit = max(suit_counts.values()) if suit_counts else 0

    bonus = 0.0
    if max_suit == 4:
        bonus += 0.16

    ranks_present = set(RANK_VALUE[c[0]] for c in all_cards)
    if 14 in ranks_present:
        ranks_present.add(1)

    straight_outs = 0
    for target in range(1, 15):
        if target not in ranks_present:
            test_set = ranks_present | {target}
            for low in range(1, 11):
                if all((low + k) in test_set for k in range(5)):
                    straight_outs += 1
                    break

    if straight_outs >= 2:
        bonus += 0.14
    elif straight_outs == 1:
        bonus += 0.06

    return bonus

def evaluate_discard(hole_cards, flop_board, dead_cards=None, trials=30):
    best_utility = -100.0
    best_discard = hole_cards[0]
    dead_set = set(hole_cards) | set(flop_board) | (set(dead_cards) if dead_cards else set())
    available = [c for c in FULL_DECK if c not in dead_set]

    for i in range(3):
        keep = [hole_cards[j] for j in range(3) if j != i]
        cand = hole_cards[i]

        made = best_hand(keep, flop_board)
        made_score = made[0] * 10 + (made[1] if len(made) > 1 else 0) * 0.5
        draws = draw_bonus(keep, flop_board) * 20.0
        kicker = max(RANK_VALUE[c[0]] for c in keep) * 0.1

        wins = 0
        for _ in range(trials):
            sample = random.sample(available, 4)
            opp = sample[:2]
            future_board = flop_board + sample[2:]
            if best_hand(keep, future_board) > best_hand(opp, future_board):
                wins += 1
        eq = wins / trials if trials > 0 else 0.5

        utility = made_score + draws + kicker + (eq * 15.0)
        if utility > best_utility:
            best_utility = utility
            best_discard = cand

    return best_discard

def preflop_score(hole):
    ranks = [c[0] for c in hole]
    suits = [c[1] for c in hole]
    values = sorted([RANK_VALUE[r] for r in ranks], reverse=True)
    score = values[0] + values[1] * 0.5 + values[2] * 0.2
    counts = Counter(values)
    if 3 in counts.values():
        score += 25
    elif 2 in counts.values():
        score += 15
    suit_counts = Counter(suits)
    if 3 in suit_counts.values():
        score += 8
    elif 2 in suit_counts.values():
        score += 4
    if values[0] - values[1] == 1 or values[1] - values[2] == 1:
        score += 3
    return score

def choose_action(req, state):
    legal = req["legal_actions"]
    street = req["street"]
    to_call = req["to_call"]
    pot = req["pot"]
    hole = state.get("hole_cards", [])
    board = req.get("board", [])

    if street == "preflop":
        score = preflop_score(hole)
        if to_call == 0:
            if score >= 20 and "raise" in legal:
                amt = min(req["max_raise"], max(req["min_raise"], 6))
                return {"type": "action", "action": "raise", "amount": amt}
            return {"type": "action", "action": "check"}
        else:
            if to_call <= 2:
                return {"type": "action", "action": "call"}
            if score >= 28 and "raise" in legal:
                amt = min(req["max_raise"], max(req["min_raise"], req["opp_bet"] * 3))
                return {"type": "action", "action": "raise", "amount": amt}
            if score >= 17:
                if to_call <= 12 or (score >= 22 and to_call <= 30):
                    return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

    hand_val = best_hand(hole, board)
    category = hand_val[0]
    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0

    if "check" in legal:
        if category >= 3 and "raise" in legal:
            target = req["opp_bet"] + int(0.6 * pot)
            return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], target))}
        elif category in (1, 2) and "raise" in legal and street in ("flop", "turn"):
            target = req["opp_bet"] + int(0.4 * pot)
            return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], target))}
        return {"type": "action", "action": "check"}

    if category >= 4:
        if "raise" in legal and req["raises_left"] > 0:
            target = req["opp_bet"] + pot
            return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], target))}
        return {"type": "action", "action": "call"}

    if category == 3:
        if to_call <= 60 or pot_odds < 0.4:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "call"}

    if category == 2:
        if to_call <= 40 or pot_odds < 0.35:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}

    if category == 1:
        if to_call <= 10 or pot_odds <= 0.22:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}

    if to_call <= 2 and pot_odds <= 0.15:
        return {"type": "action", "action": "call"}

    return {"type": "action", "action": "fold"}

def main():
    state = {"hole_cards": [], "dead_cards": set()}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        t = msg.get("type")
        if t == "new_hand":
            state["hole_cards"] = list(msg.get("hole_cards", []))
            state["dead_cards"] = set(state["hole_cards"])
        elif t == "discard_request":
            hole = msg.get("hole_cards", state["hole_cards"])
            board = msg.get("board", [])
            card = evaluate_discard(hole, board, state["dead_cards"])
            print(json.dumps({"type": "discard", "card": card}), flush=True)
        elif t == "discard_result":
            state["hole_cards"] = list(msg.get("hole_cards", []))
            state["dead_cards"].add(msg.get("my_discard"))
            if msg.get("opp_discard"):
                state["dead_cards"].add(msg.get("opp_discard"))
        elif t == "action_request":
            reply = choose_action(msg, state)
            print(json.dumps(reply), flush=True)
        elif t == "match_end":
            break

if __name__ == "__main__":
    main()
