"""End-to-end API tests: a full mock-mode match over HTTP, through to autopsy."""

import httpx
import pytest

from openswindle import fairness
from openswindle.api import app
from openswindle.models import Bid


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
    assert set(created["tokens"]) == {"a", "b"}
    match_id = created["match_id"]
    profile = await client.get(f"/matches/{match_id}/npc/profile")
    assert profile.status_code == 404

    # Seat b can act after seat a, and no NPC intervenes.
    ha = {"X-Player-Token": created["tokens"]["a"]}
    hb = {"X-Player-Token": created["tokens"]["b"]}
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
    hb = {"X-Player-Token": created["tokens"]["b"]}
    response = await client.post(
        f"/matches/{match_id}/moves",
        json={"move": {"action": "bid", "bid": {"quantity": 1, "face": 1}}},
        headers=hb,
    )
    assert response.status_code == 409
