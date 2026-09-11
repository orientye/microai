"""Public combo features. Actor must not see separated opponent hands."""

from __future__ import annotations

from collections import Counter

import numpy as np

FEAT_DIM = 25
STAT_DIM = 7
LEAD_NAMES = (
    "empty",
    "single",
    "pair",
    "triple",
    "triple_kick",
    "straight",
    "pair_straight",
    "plane",
    "bomb",
    "rocket",
    "other",
)
LEAD_DIM = len(LEAD_NAMES)
CHAIN_RANKS = tuple(range(3, 15))  # 3..A; 2 and jokers stay out


def _rank_counts(cards: list[int]) -> Counter:
    return Counter(int(c) for c in cards)


def hand_stats(cards: list[int]) -> np.ndarray:
    counts = _rank_counts(cards)
    n_single = n_pair = n_triple = n_bomb = 0
    for rank in list(range(3, 15)) + [17]:
        n = counts.get(rank, 0)
        if n == 1:
            n_single += 1
        elif n == 2:
            n_pair += 1
        elif n == 3:
            n_triple += 1
        elif n == 4:
            n_bomb += 1
    n_jokers = int(20 in counts) + int(30 in counts)
    rocket = 1.0 if (20 in counts and 30 in counts) else 0.0
    return np.array(
        [
            len(cards) / 20.0,
            n_single / 13.0,
            n_pair / 13.0,
            n_triple / 13.0,
            n_bomb / 13.0,
            n_jokers / 2.0,
            rocket,
        ],
        dtype=np.float32,
    )


def _consecutive(ranks: list[int], need: int) -> bool:
    if len(ranks) < need:
        return False
    ranks = sorted(set(ranks))
    if any(r not in CHAIN_RANKS for r in ranks):
        return False
    return ranks[-1] - ranks[0] == len(ranks) - 1


def classify_lead(cards: list[int]) -> str:
    if not cards:
        return "empty"
    counts = _rank_counts(cards)
    ranks = list(counts)
    if sorted(cards) == [20, 30]:
        return "rocket"
    if len(cards) == 4 and len(ranks) == 1:
        return "bomb"
    if len(cards) == 1:
        return "single"
    if len(cards) == 2 and len(ranks) == 1:
        return "pair"
    if len(cards) == 3 and len(ranks) == 1:
        return "triple"
    if len(cards) == 4 and sorted(counts.values()) == [1, 3]:
        return "triple_kick"
    if all(c == 1 for c in counts.values()) and _consecutive(ranks, 5):
        return "straight"
    if all(c == 2 for c in counts.values()) and _consecutive(ranks, 3):
        return "pair_straight"
    triple_ranks = [r for r, n in counts.items() if n >= 3 and r in CHAIN_RANKS]
    if len(triple_ranks) >= 2:
        triple_ranks.sort()
        for i in range(len(triple_ranks) - 1):
            if triple_ranks[i + 1] - triple_ranks[i] == 1:
                return "plane"
    return "other"


def lead_onehot(cards: list[int]) -> np.ndarray:
    out = np.zeros(LEAD_DIM, dtype=np.float32)
    out[LEAD_NAMES.index(classify_lead(list(cards)))] = 1.0
    return out


def encode_public_feat(
    hand: list[int], other_union: list[int], lead_cards: list[int]
) -> np.ndarray:
    return np.concatenate(
        [hand_stats(hand), hand_stats(other_union), lead_onehot(lead_cards)]
    ).astype(np.float32)


def feat_from_infoset(infoset) -> np.ndarray:
    return encode_public_feat(
        list(infoset.player_hand_cards),
        list(infoset.other_hand_cards),
        list(infoset.last_move),
    )
