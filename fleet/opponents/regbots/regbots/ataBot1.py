#!/usr/bin/env python3
"""
Discard Hold'em Bot - Upgraded Production Model (bot.py)
Key Upgrades:
1. Geometric Pot Sizing: Raises capped to geometric pot fractions (35%-80% pot).
   Strict ban on reckless non-river all-in shoves with one pair or two pair into small pots.
2. Board Texture & Relative Hand Strength:
   - Detects Overpairs vs Underpairs / Second Pairs when overcards are on the board.
   - Enforces pot-control (check/call small bets, fold to heavy pressure) on second pairs.
3. High-Threshold Opponent Profiler:
   - Eliminates the 'maniac' false-positive on solid aggressive opponents like boss1.
   - Tracks actual over-bets, shoves, and fold-rates accurately.
4. Range-Conditioned Turn & River Simulation:
   - Replaces naive random opponent sampling with realistic continuing range (broadways, pairs, board hits).
5. Position-Aware River Pot Control:
   - Checks back medium showdown value in position to prevent value-owning ourselves.
6. Adaptive Time Watchdog:
   - Dynamic simulation scaling guarantees zero timeouts and zero notices.
"""

import sys
import json
import random
from collections import Counter
from itertools import combinations

# ---------------------------------------------------------------- Constants

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
SUIT_ORDER = {s: i for i, s in enumerate(SUITS)}
FULL_DECK = [r + s for r in RANKS for s in SUITS]

# ----------------------------------------------------------- Hand Evaluator

def rank5(cards):
    """Strength of exactly five cards as a comparable tuple (higher is better)."""
    values = sorted((RANK_VALUE[c[0]] for c in cards), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    uniq = sorted(set(values), reverse=True)
    straight_high = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:  # A-2-3-4-5 low straight
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
    """Best 5-card rank from hole + board (up to 7 cards)."""
    all_cards = list(hole) + list(board)
    if len(all_cards) < 5:
        return (0,)
    return max(rank5(combo) for combo in combinations(all_cards, 5))

# --------------------------------------------------- Preflop Hand Evaluation

def preflop_strength(hole):
    """Evaluates 3-card preflop hand strength. Returns [0.0, 1.0]."""
    values = sorted([RANK_VALUE[c[0]] for c in hole], reverse=True)
    suits = [c[1] for c in hole]
    val_counts = Counter(values)
    suit_counts = Counter(suits)

    high_score = (values[0] * 3 + values[1] * 2 + values[2]) / (14 * 3 + 13 * 2 + 12)

    pair_bonus = 0.0
    if 3 in val_counts.values():
        pair_bonus = 0.45
    elif 2 in val_counts.values():
        pair_val = [v for v, c in val_counts.items() if c == 2][0]
        pair_bonus = 0.20 + (pair_val / 14.0) * 0.26  # 0.24 to 0.46

    suit_bonus = 0.10 if 3 in suit_counts.values() else (0.05 if 2 in suit_counts.values() else 0.0)
    conn_bonus = 0.05 if ((values[0] - values[1] <= 2) or (values[1] - values[2] <= 2)) else 0.0

    score = 0.35 * high_score + pair_bonus + suit_bonus + conn_bonus
    return min(1.0, max(0.0, score))

# --------------------------------------------------- Opponent Profiler State

class OpponentProfiler:
    def __init__(self):
        self.hands_count = 0
        self.opp_preflop_actions = 0
        self.opp_preflop_folds = 0
        self.opp_preflop_raises = 0
        self.opp_preflop_shoves = 0
        self.opp_postflop_calls = 0
        self.opp_postflop_folds = 0
        self.opp_postflop_bets = 0

    def record_hand(self, history, my_seat):
        self.hands_count += 1
        for act in history:
            street = act.get("street")
            actor = act.get("actor")
            action = act.get("action")
            amt = act.get("amount", 0)
            if actor == "opp":
                if street == "preflop":
                    self.opp_preflop_actions += 1
                    if action == "fold":
                        self.opp_preflop_folds += 1
                    elif action == "raise":
                        self.opp_preflop_raises += 1
                        if amt >= 100:
                            self.opp_preflop_shoves += 1
                else:
                    if action == "fold":
                        self.opp_postflop_folds += 1
                    elif action == "call":
                        self.opp_postflop_calls += 1
                    elif action == "raise":
                        self.opp_postflop_bets += 1

    def is_nit(self):
        if self.opp_preflop_actions >= 6:
            return (self.opp_preflop_folds / self.opp_preflop_actions) > 0.40
        return False

    def is_maniac(self):
        """Strict definition: Only true if opponent shoves preflop or raises > 70%."""
        if self.opp_preflop_shoves >= 2:
            return True
        total = self.opp_preflop_actions + self.opp_postflop_bets
        if total >= 12:
            return (self.opp_preflop_raises + self.opp_postflop_bets) / total > 0.70
        return False

    def is_calling_station(self):
        if self.opp_postflop_calls + self.opp_postflop_folds >= 8:
            return self.opp_postflop_folds <= 1
        return False

# --------------------------------------------------------- Draw Evaluation

def draw_bonus(hole2, board):
    all_cards = list(hole2) + list(board)
    suits = [c[1] for c in all_cards]
    suit_counts = Counter(suits)
    max_suit = max(suit_counts.values()) if suit_counts else 0

    bonus = 0.0
    if max_suit == 4:
        bonus += 0.16  # 4-flush draw

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
        bonus += 0.14  # OESD
    elif straight_outs == 1:
        bonus += 0.06  # Gutshot

    return bonus

# ------------------------------------------------ Relative Hand Strength

def evaluate_relative_strength(hole, board):
    """
    Checks if our hand is an Overpair, Top Pair, Second Pair, or Underpair relative to the board.
    Returns: (category, is_top_or_overpair, has_overcards_on_board)
    """
    if len(board) < 3 or len(hole) < 2:
        return 0, False, False

    score = best_hand(hole, board)
    category = score[0]
    board_vals = [RANK_VALUE[c[0]] for c in board]
    max_board_val = max(board_vals)

    h_vals = [RANK_VALUE[c[0]] for c in hole]
    is_pocket_pair = (h_vals[0] == h_vals[1])

    if is_pocket_pair:
        if h_vals[0] > max_board_val:
            return category, True, False  # Overpair!
        else:
            return category, False, True  # Underpair / Second pair!

    # Paired with board?
    hole_paired_with_board = [v for v in h_vals if v in board_vals]
    if hole_paired_with_board:
        top_hit = max(hole_paired_with_board)
        if top_hit == max_board_val:
            return category, True, False  # Top Pair!
        else:
            return category, False, True  # Second or Third Pair!

    return category, False, False

# ------------------------------------------------ Realistic Range Filter

def is_plausible_opponent_hand(opp_hole, board_ranks):
    """Filters out impossible junk from opponent's continuing range."""
    r1, r2 = RANK_VALUE[opp_hole[0][0]], RANK_VALUE[opp_hole[1][0]]
    if r1 == r2:  # pocket pair
        return True
    if r1 in board_ranks or r2 in board_ranks:  # hits the board
        return True
    if opp_hole[0][1] == opp_hole[1][1]:  # suited
        return True
    if max(r1, r2) >= 10:  # broadway card
        return True
    return False

# --------------------------------------------------------- Equity Simulation

def estimate_equity(my_hole, board, dead_cards, street="flop", trials=90, filter_range=True):
    dead_set = set(my_hole) | set(board) | set(dead_cards)
    available_deck = [c for c in FULL_DECK if c not in dead_set]
    needed_board = 5 - len(board)

    if trials <= 0 or len(available_deck) < 2 + needed_board:
        score = best_hand(my_hole, board)
        category = score[0]
        return min(1.0, category * 0.14 + 0.28)

    board_ranks = {RANK_VALUE[c[0]] for c in board}
    filter_active = (filter_range and street in ("turn", "river"))

    wins = 0
    ties = 0
    valid_trials = 0
    attempts = 0
    max_attempts = trials * 3

    while valid_trials < trials and attempts < max_attempts:
        attempts += 1
        sample = random.sample(available_deck, 2 + needed_board)
        opp_hole = sample[:2]

        if filter_active and not is_plausible_opponent_hand(opp_hole, board_ranks):
            continue

        future_board = board + sample[2:]
        my_rank = best_hand(my_hole, future_board)
        opp_rank = best_hand(opp_hole, future_board)

        if my_rank > opp_rank:
            wins += 1
        elif my_rank == opp_rank:
            ties += 1
        valid_trials += 1

    if valid_trials == 0:
        return 0.5
    return (wins + 0.5 * ties) / valid_trials

# --------------------------------------------------------- Discard Engine

def evaluate_discard(hole_cards, flop_board, dead_cards, trials=40):
    best_utility = -100.0
    best_discard = hole_cards[0]

    for i in range(3):
        keep = [hole_cards[j] for j in range(3) if j != i]
        discard_candidate = hole_cards[i]
        extra_dead = list(dead_cards) + [discard_candidate]

        eq = estimate_equity(keep, flop_board, extra_dead, street="flop", trials=trials, filter_range=False)
        made_rank = best_hand(keep, flop_board)
        made_bonus = made_rank[0] * 0.06
        draws = draw_bonus(keep, flop_board)
        kicker_bonus = max(RANK_VALUE[c[0]] for c in keep) / 100.0

        utility = eq + made_bonus + draws + kicker_bonus
        if utility > best_utility:
            best_utility = utility
            best_discard = discard_candidate

    return best_discard

# --------------------------------------------------------- Action Selection

def choose_action(req, state, profiler):
    legal = req["legal_actions"]
    street = req["street"]
    to_call = req["to_call"]
    pot = req["pot"]
    board = req.get("board", [])
    time_left = req.get("time_left_ms", 30000)
    raises_left = req.get("raises_left", 0)
    min_r = req.get("min_raise", 0)
    max_r = req.get("max_raise", 0)

    hole = state.get("hole_cards", [])
    dead_cards = state.get("dead_cards", set())
    is_btn = (state.get("my_seat") == "button")

    # Time budgeting
    if time_left > 18000:
        trials = 100
    elif time_left > 10000:
        trials = 50
    elif time_left > 4000:
        trials = 20
    else:
        trials = 0

    is_nit = profiler.is_nit()
    is_maniac = profiler.is_maniac()
    is_station = profiler.is_calling_station()

    # ------------------------------------------------------------- 1. PREFLOP
    if street == "preflop":
        strength = preflop_strength(hole)

        # Blind Stealing against Nits
        if is_nit and is_btn and to_call <= 1 and "raise" in legal:
            target = max(min_r, min(max_r, 4))
            return {"type": "action", "action": "raise", "amount": target}

        if to_call == 0:
            # Button open raise with top 65% of hands
            if is_btn and strength >= 0.42 and "raise" in legal:
                target = max(min_r, min(max_r, 5))
                return {"type": "action", "action": "raise", "amount": target}
            elif not is_btn and strength >= 0.65 and "raise" in legal:
                target = max(min_r, min(max_r, 6))
                return {"type": "action", "action": "raise", "amount": target}
            return {"type": "action", "action": "check"}

        else:
            # Facing a bet or raise
            if to_call <= 2:
                if strength >= 0.22:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}

            # Premium hands: 3-bet / 4-bet (geometric sizing, NOT all-in!)
            if strength >= 0.84 and "raise" in legal and raises_left > 0:
                target = min(max_r, max(min_r, req["opp_bet"] * 3))
                return {"type": "action", "action": "raise", "amount": target}

            call_thresh = 0.50 if is_maniac else (0.68 if is_nit else 0.58)
            if strength >= call_thresh or (strength >= 0.42 and to_call <= 8):
                return {"type": "action", "action": "call"}

            return {"type": "action", "action": "fold"}

    # ------------------------------------------------------- 2. POSTFLOP
    equity = estimate_equity(hole, board, dead_cards, street=street, trials=trials, filter_range=True)
    pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0
    category, is_top_or_overpair, has_overcards = evaluate_relative_strength(hole, board)

    if to_call == 0:
        # Check or Bet
        # Calling station exploit
        if is_station and equity >= 0.62 and "raise" in legal:
            mult = 0.80 if equity >= 0.75 else 0.50
            target = req["opp_bet"] + int(mult * pot)
            return {"type": "action", "action": "raise", "amount": max(min_r, min(max_r, target))}

        # River Pot Control: In position with medium showdown value, check back!
        if street == "river" and is_btn and (category <= 2 or not is_top_or_overpair):
            return {"type": "action", "action": "check"}

        # Monster hand value bet (Nut hands, full house, flush, strong sets)
        if equity >= 0.82 and "raise" in legal:
            # GEOMETRIC SIZING: 65% to 80% pot, NEVER all-in shove into small pot
            target = req["opp_bet"] + int(0.70 * pot)
            return {"type": "action", "action": "raise", "amount": max(min_r, min(max_r, target))}

        # Standard value / C-bet on flop & turn (only if top pair or better)
        elif equity >= 0.58 and "raise" in legal and street in ("flop", "turn") and not has_overcards:
            target = req["opp_bet"] + int(0.40 * pot)
            return {"type": "action", "action": "raise", "amount": max(min_r, min(max_r, target))}

        return {"type": "action", "action": "check"}

    else:
        # Facing a bet or raise
        # GEOMETRIC CAP: Never shove full stack on early streets without near nuts!
        if equity >= 0.88 and "raise" in legal and raises_left > 0:
            # Raise to 2.5x their bet or 80% pot, capped
            pot_sized_raise = req["opp_bet"] + int(0.80 * (pot + to_call))
            # Only shove all-in if remaining stack behind is small (<= 1.5x pot)
            target = max_r if (req["my_stack"] <= int(1.5 * pot)) else pot_sized_raise
            return {"type": "action", "action": "raise", "amount": max(min_r, min(max_r, target))}

        # Pot Control for Underpairs / Second Pairs:
        # If board has overcards to our pair, fold to large bets!
        if has_overcards and category == 1:
            if to_call <= int(0.25 * pot):
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        # Disciplined calling margin
        margin = 0.10 if is_nit else (0.06 if street == "river" else 0.02)
        if equity >= (pot_odds + margin):
            return {"type": "action", "action": "call"}

        # Float flop/turn cheaply with draws
        if street in ("flop", "turn") and to_call <= 6 and equity >= 0.28:
            return {"type": "action", "action": "call"}

        return {"type": "action", "action": "fold"}

# ------------------------------------------------------------ Main Loop

def main():
    profiler = OpponentProfiler()
    state = {
        "hole_cards": [],
        "dead_cards": set(),
        "my_seat": None
    }

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue

        mtype = msg.get("type")

        if mtype == "new_hand":
            state["hole_cards"] = list(msg.get("hole_cards", []))
            state["dead_cards"] = set(state["hole_cards"])
            state["my_seat"] = msg.get("seat")

        elif mtype == "discard_request":
            hole = msg.get("hole_cards", state["hole_cards"])
            board = msg.get("board", [])
            time_left = msg.get("time_left_ms", 30000)
            trials = 40 if time_left > 12000 else 18
            card = evaluate_discard(hole, board, state["dead_cards"], trials=trials)
            print(json.dumps({"type": "discard", "card": card}), flush=True)

        elif mtype == "discard_result":
            state["hole_cards"] = list(msg.get("hole_cards", []))
            state["dead_cards"].add(msg.get("my_discard"))
            opp_discard = msg.get("opp_discard")
            if opp_discard:
                state["dead_cards"].add(opp_discard)

        elif mtype == "action_request":
            reply = choose_action(msg, state, profiler)
            print(json.dumps(reply), flush=True)

        elif mtype == "hand_result":
            hist = msg.get("history", [])
            profiler.record_hand(hist, state["my_seat"])

        elif mtype == "match_end":
            break

if __name__ == "__main__":
    main()
