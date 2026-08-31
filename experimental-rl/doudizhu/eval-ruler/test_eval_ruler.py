"""Tests for the Dou Dizhu eval ruler (fixed decks + seat swap + WP/ADP)."""

from generate_eval_data import generate_deals
from eval_ruler import aggregate_seat_swap, paper_adp, play_deals, resolve_roles


def test_paper_adp_no_bomb():
    assert paper_adp(landlord_won=True, bomb_num=0) == 1
    assert paper_adp(landlord_won=False, bomb_num=0) == -1


def test_paper_adp_two_bombs():
    assert paper_adp(landlord_won=True, bomb_num=2) == 4
    assert paper_adp(landlord_won=False, bomb_num=2) == -4


def test_seat_swap_always_landlord_wins_is_even():
    seating_a_landlord = [(True, 0)] * 10
    seating_b_landlord = [(True, 0)] * 10
    out = aggregate_seat_swap(seating_a_landlord, seating_b_landlord)
    assert out["games"] == 20
    assert out["wp_a"] == 0.5
    assert out["adp_a"] == 0.0


def test_seat_swap_a_wins_every_role():
    seating_a_landlord = [(True, 0)] * 4
    seating_b_landlord = [(False, 0)] * 4
    out = aggregate_seat_swap(seating_a_landlord, seating_b_landlord)
    assert out["wp_a"] == 1.0
    assert out["adp_a"] == 1.0


def test_generate_deals_counts_and_seed():
    deals = generate_deals(num_games=5, seed=0)
    assert len(deals) == 5
    for deal in deals:
        assert len(deal["landlord"]) == 20
        assert len(deal["landlord_up"]) == 17
        assert len(deal["landlord_down"]) == 17
        all_cards = (
            deal["landlord"] + deal["landlord_up"] + deal["landlord_down"]
        )
        assert len(all_cards) == 54
        assert sorted(all_cards) == sorted(
            [i for i in range(3, 15) for _ in range(4)] + [17] * 4 + [20, 30]
        )
        assert len(deal["three_landlord_cards"]) == 3
        from collections import Counter

        leftover = Counter(deal["three_landlord_cards"]) - Counter(
            deal["landlord"]
        )
        assert not leftover

    again = generate_deals(num_games=5, seed=0)
    assert deals == again
    other = generate_deals(num_games=5, seed=1)
    assert deals != other


def test_play_deals_does_not_mutate_hands():
    deals = generate_deals(num_games=2, seed=0)
    before = [list(d["landlord"]) for d in deals]
    play_deals(deals, "random", "random", "random")
    after = [list(d["landlord"]) for d in deals]
    assert before == after
    play_deals(deals, "random", "random", "random")
    after2 = [list(d["landlord"]) for d in deals]
    assert before == after2


def test_resolve_roles_random():
    assert resolve_roles("random") == {
        "landlord": "random",
        "landlord_up": "random",
        "landlord_down": "random",
    }


def test_play_deals_with_players_two_deals():
    from eval_ruler import _load_players, play_deals_with_players

    deals = generate_deals(num_games=2, seed=0)
    before = [list(d["landlord"]) for d in deals]
    players = _load_players("random", "random", "random")
    out = play_deals_with_players(deals, players)
    assert len(out) == 2
    assert all(isinstance(won, bool) and k >= 0 for won, k in out)
    assert [list(d["landlord"]) for d in deals] == before


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("all passed")
