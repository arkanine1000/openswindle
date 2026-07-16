from openswindle import fairness


def test_deal_is_deterministic_and_mutual():
    salt_a = b"a" * 32
    salt_b = b"b" * 32
    hand1 = fairness.deal_hand(salt_a, salt_b, 1, "a", 4)
    hand2 = fairness.deal_hand(salt_a, salt_b, 1, "a", 4)
    assert hand1 == hand2
    assert all(1 <= d <= 4 for d in hand1)

    # Changing either salt changes the outcome distribution source.
    other = fairness.deal_hand(b"c" * 32, salt_b, 1, "a", 4)
    another = fairness.deal_hand(salt_a, b"c" * 32, 1, "a", 4)
    assert other != hand1 or another != hand1


def test_hands_differ_by_seat_and_round():
    salt_a, salt_b = b"a" * 32, b"b" * 32
    assert fairness.deal_hand(salt_a, salt_b, 1, "a", 6) != fairness.deal_hand(
        salt_a, salt_b, 1, "b", 6
    ) or fairness.deal_hand(salt_a, salt_b, 2, "a", 6) != fairness.deal_hand(
        salt_a, salt_b, 1, "a", 6
    )


def test_commit_reveal_round_trip():
    salt = fairness.draw_salt()
    hand = [3, 1, 4, 2]
    commitment = fairness.commit_hand(salt, hand)
    assert fairness.verify_commitment(salt.hex(), hand, commitment)
    # Order-insensitive: commitments are over the sorted hand.
    assert fairness.verify_commitment(salt.hex(), sorted(hand), commitment)
    # Tampered hand fails.
    assert not fairness.verify_commitment(salt.hex(), [1, 1, 1, 1], commitment)
    # Wrong salt fails.
    assert not fairness.verify_commitment(fairness.draw_salt().hex(), hand, commitment)


def test_salts_are_unique():
    assert fairness.draw_salt() != fairness.draw_salt()
