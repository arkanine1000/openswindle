"""Cryptographic fairness: mutual-entropy dealing and SHA-256 commit-reveal.

Protocol
--------
At the start of every round the server draws a fresh 32-byte salt per hand.
Die ``i`` of the hand for ``seat`` is derived from *both* salts so neither
party's entropy alone determines the outcome:

    die_i = (int(sha256(salt_a || salt_b || round_no || seat || i)) % 4) + 1

The server immediately publishes a commitment per hand:

    commitment = sha256(salt_seat || hand_bytes)

where ``hand_bytes`` is the canonical byte string of sorted die faces.
On any round termination (call or abort) both salts and hands are revealed
unconditionally, letting any client re-derive the dice and verify the
commitments. Salting per hand (not per round) blocks dictionary attacks
against the small hand space.
"""

import hashlib
import secrets

from .models import Seat

SALT_BYTES = 32


def draw_salt() -> bytes:
    return secrets.token_bytes(SALT_BYTES)


def _die(salt_a: bytes, salt_b: bytes, round_no: int, seat: Seat, index: int) -> int:
    material = (
        salt_a
        + salt_b
        + round_no.to_bytes(4, "big")
        + seat.encode()
        + index.to_bytes(4, "big")
    )
    digest = hashlib.sha256(material).digest()
    return (int.from_bytes(digest, "big") % 4) + 1


def deal_hand(salt_a: bytes, salt_b: bytes, round_no: int, seat: Seat, count: int) -> list[int]:
    """Derive a hand of ``count`` d4 dice from both salts (mutual entropy)."""
    return sorted(_die(salt_a, salt_b, round_no, seat, i) for i in range(count))


def canonical_hand_bytes(hand: list[int]) -> bytes:
    return bytes(sorted(hand))


def commit_hand(salt: bytes, hand: list[int]) -> str:
    """SHA-256 commitment H(salt || hand) published at deal time."""
    return hashlib.sha256(salt + canonical_hand_bytes(hand)).hexdigest()


def verify_commitment(salt_hex: str, hand: list[int], commitment: str) -> bool:
    """Client-side audit helper: check a revealed salt+hand against its commitment."""
    return commit_hand(bytes.fromhex(salt_hex), hand) == commitment
