"""End-to-end API tests: a full mock-mode match over HTTP, through to autopsy."""

import httpx
import pytest

from openswindle import fairness
from openswindle.api import app
from openswindle.models import Bid
from openswindle.store import store


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def create_match(client: httpx.AsyncClient, **config) -> dict:
    response = await client.post("/matches", json={"config": config})
    assert response.status_code == 200, response.text
    return response.json()


def raise_over(view: dict) -> dict:
    """A guaranteed-legal raise from the current public view."""
    history = view["bid_history"]
    if not history:
        return {"quantity": 1, "face": 1}
    last = Bid.model_validate(history[-1]["bid"])
    total = sum(view["dice_counts"].values())
    if last.face < 4:
        return {"quantity": last.quantity, "face": last.face + 1}
    if last.quantity < total:
        return {"quantity": last.quantity + 1, "face": 1}
    return {}  # No raise possible: caller must call.


async def test_healthz(client):
    response = await client.get("/healthz")
    assert response.status_code == 200


async def test_full_match_to_autopsy(client):
    created = await create_match(
        client, dice_per_player=2, opponent_type="llm", npc_seed="seed 4471"
    )
    match_id = created["match_id"]
    token = created["tokens"]["a"]
    headers = {"X-Player-Token": token}
    assert created["npc_name"]
    view = created["view"]
    assert view["your_hand"] and len(view["your_hand"]) == 2
    assert set(view["commitments"]) == {"a", "b"}

    # Autopsy is locked while the match runs.
    locked = await client.get(f"/matches/{match_id}/autopsy")
    assert locked.status_code == 409

    for _ in range(200):
        if view["phase"] == "finished":
            break
        if view["turn"] != "a":
            refreshed = await client.get(f"/matches/{match_id}", headers=headers)
            view = refreshed.json()
            continue
        bid = raise_over(view)
        move = {"action": "bid", "bid": bid} if bid else {"action": "call"}
        response = await client.post(
            f"/matches/{match_id}/moves",
            json={"move": move, "table_talk": "I never lie."},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        body = response.json()
        view = body["view"]
        # Every reveal must verify against its commitments.
        for reveal in body["reveals"]:
            for seat in ("a", "b"):
                assert fairness.verify_commitment(
                    reveal["salts"][seat],
                    reveal["hands"][seat],
                    reveal["commitments"][seat],
                )
    assert view["phase"] == "finished"
    assert view["winner"] in ("a", "b")

    autopsy = await client.get(f"/matches/{match_id}/autopsy")
    assert autopsy.status_code == 200
    body = autopsy.json()
    assert body["npc_profile"]["params"]
    assert body["decisions"], "NPC decisions must be recorded"
    for decision in body["decisions"]:
        assert decision["deviation_price"] >= -1e-9
        assert decision["scratchpad"]


async def test_abort_reveals_current_round_and_finishes(client):
    created = await create_match(client, dice_per_player=2, opponent_type="llm")
    match_id = created["match_id"]
    headers = {"X-Player-Token": created["tokens"]["a"]}

    response = await client.post(f"/matches/{match_id}/abort", headers=headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["view"]["phase"] == "finished"
    assert len(body["reveals"]) == 1
    reveal = body["reveals"][0]
    for seat in ("a", "b"):
        assert fairness.verify_commitment(
            reveal["salts"][seat],
            reveal["hands"][seat],
            reveal["commitments"][seat],
        )

    autopsy = await client.get(f"/matches/{match_id}/autopsy")
    assert autopsy.status_code == 200
    second_abort = await client.post(f"/matches/{match_id}/abort", headers=headers)
    assert second_abort.status_code == 409


async def test_auth_required(client):
    created = await create_match(client, dice_per_player=2)
    match_id = created["match_id"]
    assert (await client.get(f"/matches/{match_id}")).status_code == 401
    bad = {"X-Player-Token": "nope"}
    assert (await client.get(f"/matches/{match_id}", headers=bad)).status_code == 403


async def test_tokens_are_scoped_to_their_match(client):
    first = await create_match(client, dice_per_player=2)
    second = await create_match(client, dice_per_player=2)
    cross = {"X-Player-Token": first["tokens"]["a"]}
    response = await client.get(f"/matches/{second['match_id']}", headers=cross)
    assert response.status_code == 403


async def test_illegal_move_rejected(client):
    created = await create_match(client, dice_per_player=2)
    match_id = created["match_id"]
    headers = {"X-Player-Token": created["tokens"]["a"]}
    # Calling before any bid is illegal.
    response = await client.post(
        f"/matches/{match_id}/moves", json={"move": {"action": "call"}}, headers=headers
    )
    assert response.status_code == 400


async def test_human_vs_human_two_tokens_no_npc(client):
    created = await create_match(client, dice_per_player=2, opponent_type="human")
    assert set(created["tokens"]) == {"a"}
    match_id = created["match_id"]
    profile = await client.get(f"/matches/{match_id}/npc/profile")
    assert profile.status_code == 404

    joined = await client.post(f"/matches/{match_id}/join")
    assert joined.status_code == 200
    joined_body = joined.json()
    assert joined_body["seat"] == "b"
    assert joined_body["token"]
    assert (await client.post(f"/matches/{match_id}/join")).status_code == 409

    # Seat b can act after seat a, but receives its token separately and no NPC intervenes.
    ha = {"X-Player-Token": created["tokens"]["a"]}
    hb = {"X-Player-Token": joined_body["token"]}
    r1 = await client.post(
        f"/matches/{match_id}/moves",
        json={"move": {"action": "bid", "bid": {"quantity": 1, "face": 1}}},
        headers=ha,
    )
    assert r1.status_code == 200
    assert r1.json()["npc_events"] == []
    r2 = await client.post(
        f"/matches/{match_id}/moves",
        json={"move": {"action": "bid", "bid": {"quantity": 1, "face": 2}}},
        headers=hb,
    )
    assert r2.status_code == 200


async def test_wrong_turn_conflict(client):
    created = await create_match(client, dice_per_player=2, opponent_type="human")
    match_id = created["match_id"]
    joined = await client.post(f"/matches/{match_id}/join")
    hb = {"X-Player-Token": joined.json()["token"]}
    response = await client.post(
        f"/matches/{match_id}/moves",
        json={"move": {"action": "bid", "bid": {"quantity": 1, "face": 1}}},
        headers=hb,
    )
    assert response.status_code == 409


async def test_scripted_telemetry_marks_susceptibility_off(client):
    created = await create_match(client, dice_per_player=2, opponent_type="scripted")
    match_id = created["match_id"]
    headers = {"X-Player-Token": created["tokens"]["a"]}
    response = await client.post(
        f"/matches/{match_id}/moves",
        json={
            "move": {"action": "bid", "bid": {"quantity": 1, "face": 1}},
            "table_talk": "please believe me",
        },
        headers=headers,
    )
    assert response.status_code == 200, response.text

    await client.post(f"/matches/{match_id}/abort", headers=headers)
    autopsy = await client.get(f"/matches/{match_id}/autopsy")
    assert autopsy.status_code == 200
    decisions = autopsy.json()["decisions"]
    assert decisions
    assert all(not decision["susceptibility_on"] for decision in decisions)
    assert all(decision["human_table_talk_seen"] is None for decision in decisions)


async def test_transcript_logs_move_before_talk_and_seat_free_reveals(client):
    created = await create_match(
        client, dice_per_player=2, opponent_type="llm", npc_seed="seed 4471"
    )
    match_id = created["match_id"]
    headers = {"X-Player-Token": created["tokens"]["a"]}
    view = created["view"]
    for _ in range(200):
        if view["phase"] == "finished":
            break
        if view["turn"] != "a":
            refreshed = await client.get(f"/matches/{match_id}", headers=headers)
            view = refreshed.json()
            continue
        bid = raise_over(view)
        move = {"action": "bid", "bid": bid} if bid else {"action": "call"}
        response = await client.post(
            f"/matches/{match_id}/moves",
            json={"move": move, "table_talk": "watch closely"},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        view = response.json()["view"]
    assert view["phase"] == "finished"

    transcript = store.get(match_id).transcript
    # Table talk is said as the move is made: it must directly follow its move.
    talk_indices = [i for i, e in enumerate(transcript) if e.kind == "talk" and e.seat == "a"]
    assert talk_indices
    for i in talk_indices:
        assert transcript[i - 1].kind in ("bid", "call")
        assert transcript[i - 1].seat == "a"
    # Reveal events carry the loser in seat and never leak seat labels in text.
    reveals = [e for e in transcript if e.kind == "reveal"]
    assert reveals
    assert [e.seat for e in reveals] == [r["loser"] for r in view["reveals"]]
    assert all("seat" not in e.text for e in reveals)


async def test_opponent_present_flips_when_seat_b_joins(client):
    created = await create_match(client, dice_per_player=2, opponent_type="human")
    match_id = created["match_id"]
    # Seat A is alone at the table until someone claims seat B.
    assert created["view"]["opponent_present"] is False

    joined = await client.post(f"/matches/{match_id}/join")
    assert joined.status_code == 200
    assert joined.json()["view"]["opponent_present"] is True

    # Seat A's own polled view now reflects the arrival.
    a_view = await client.get(
        f"/matches/{match_id}", headers={"X-Player-Token": created["tokens"]["a"]}
    )
    assert a_view.json()["opponent_present"] is True


async def test_opponent_present_true_for_npc_matches(client):
    created = await create_match(client, dice_per_player=2, opponent_type="scripted")
    # An NPC is always at the table; seat B is never issued but is never "absent".
    assert created["view"]["opponent_present"] is True


async def test_call_table_talk_reaches_the_opponent(client):
    created = await create_match(client, dice_per_player=2, opponent_type="human")
    match_id = created["match_id"]
    ha = {"X-Player-Token": created["tokens"]["a"]}
    joined = await client.post(f"/matches/{match_id}/join")
    hb = {"X-Player-Token": joined.json()["token"]}

    await client.post(
        f"/matches/{match_id}/moves",
        json={"move": {"action": "bid", "bid": {"quantity": 1, "face": 1}}},
        headers=ha,
    )
    called = await client.post(
        f"/matches/{match_id}/moves",
        json={"move": {"action": "call"}, "table_talk": "I don't believe you."},
        headers=hb,
    )
    assert called.status_code == 200, called.text

    # Seat A polls its view and reads the caller's parting words off the reveal.
    a_view = (await client.get(f"/matches/{match_id}", headers=ha)).json()
    assert a_view["reveals"][-1]["table_talk"] == "I don't believe you."
