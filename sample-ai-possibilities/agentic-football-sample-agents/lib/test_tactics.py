"""Tests for tactics.py and the opt-in options of create_invoke_handler.

Run directly:  python test_tactics.py
"""

import asyncio
import copy
import json

from test_helpers import GAME_STATE, mock_agentcore, _FakeApp

mock_agentcore()

from agent_base import create_invoke_handler
from fallback import FWD1_CONFIG, build_fallback
from tactics import (
    ALA_DEPTH, DROP_LIMIT, PIVOT_DEPTH, TUCK_IN_Y, _last_possession,
    ala_recovery, choose_shot, fast_path, tactical_insights,
)

HOME, AWAY = 0, 1


def _with_holder(team_code: str, idx: int, pos: dict) -> dict:
    gs = copy.deepcopy(GAME_STATE)
    gs["ball"]["possessionAgentId"] = f"agentId_{idx}"
    gs["ball"]["position"] = dict(pos)
    for p in gs["players"]:
        if p["teamCode"] == team_code and p["agentId"] == f"agentId_{idx}":
            p["position"] = dict(pos)
    # Keep only the holder's team claiming that agentId, so possession is unambiguous.
    if team_code == "away":
        gs["players"] = [p for p in gs["players"]
                         if not (p["teamCode"] == "home" and p["agentId"] == f"agentId_{idx}")]
    return gs


def test_fast_path_shoots_close_to_goal():
    gs = _with_holder("home", 3, {"x": 45, "y": 2})
    cmds = fast_path(gs, HOME, 3, "FWD1")
    assert cmds and cmds[0]["commandType"] == "SHOOT", cmds
    assert cmds[0]["playerId"] == 3 and cmds[0]["teamId"] == HOME
    print(f"=== FAST PATH: close-range shot {cmds[0]['parameters']} ===")


def test_fast_path_defers_to_llm_when_far():
    """The fixture forward holds the ball 40 units out — a judgement call for the LLM."""
    assert fast_path(GAME_STATE, HOME, 3, "FWD1") is None
    print("=== FAST PATH: far from goal, LLM decides ===")


def test_fast_path_ignores_players_without_ball():
    assert fast_path(GAME_STATE, HOME, 2, "MID") is None
    print("=== FAST PATH: no ball, LLM decides ===")


def test_keeper_distributes():
    gs = _with_holder("home", 0, {"x": -50, "y": 0})
    cmds = fast_path(gs, HOME, 0, "GK")
    assert cmds and cmds[0]["commandType"] == "GK_DISTRIBUTE", cmds
    assert cmds[0]["parameters"]["target_player_id"] != 0
    print(f"=== FAST PATH: keeper distributes {cmds[0]['parameters']} ===")


def test_aim_avoids_keeper_side_for_both_teams():
    """Keeper shading +y: HOME shoots to -y (its right), AWAY to -y (its left)."""
    keeper = {"position": {"x": 50, "y": 3}}
    aim_home, _ = choose_shot({"x": 42, "y": 0}, keeper, HOME, 55.0)
    keeper_away = {"position": {"x": -50, "y": 3}}
    aim_away, _ = choose_shot({"x": -42, "y": 0}, keeper_away, AWAY, -55.0)
    assert aim_home == "BR", aim_home
    assert aim_away == "BL", aim_away
    print(f"=== AIM: home {aim_home}, away {aim_away} ===")


def test_insights_mention_plan_and_shot():
    out = tactical_insights(GAME_STATE, HOME, 3, "FWD1")
    assert "LEADING by 1" in out
    assert "Shot quality=" in out and "Best pass: player" in out
    out_away = tactical_insights(GAME_STATE, AWAY, 2, "MID")
    assert "TRAILING by 1" in out_away
    print("=== INSIGHTS: plan follows the score from each side ===")


def test_insights_track_back_without_ball():
    """Away midfielder, home attacking in away's half: not closest, so recover and mark."""
    _last_possession.clear()
    out = tactical_insights(GAME_STATE, AWAY, 2, "MID")
    assert "Ball is in OUR half" in out, out
    assert "do not chase" in out and "Recovery position" in out, out
    assert "Opponent to mark: P" in out, out
    print("=== INSIGHTS: player without the ball is told to recover and mark ===")


def test_pivot_holds_halfway_line():
    """Away pivot does not drop past x=0 even though the ball is deep in its half."""
    _last_possession.clear()
    out = tactical_insights(GAME_STATE, AWAY, 3, "FWD1")
    rec = out.split("Recovery position (goal-side of the ball): (")[1].split(")")[0]
    x = float(rec.split(",")[0])
    assert x <= 0.0, out
    assert "Opponent to mark" not in out, "pivot should not follow anyone into our half"
    print(f"=== PIVOT: recovery x={x} stays at the halfway line ===")


def test_pivot_receives_high():
    """Home pivot, teammate on the ball: open space is about PIVOT_DEPTH from goal."""
    gs = _with_holder("home", 2, {"x": 5, "y": -8})
    out = tactical_insights(gs, HOME, 3, "FWD1")
    spot = out.split("Open space to receive: (")[1].split(")")[0]
    x = float(spot.split(",")[0])
    assert 55 - PIVOT_DEPTH - 5 <= x <= 55 - PIVOT_DEPTH + 5, out
    print(f"=== PIVOT: receives high at x={x} ===")


def test_ala_tucks_in_on_weak_side():
    """Away right ala (+y), ball on the -y side: tuck in to the centre, not mark wide."""
    _last_possession.clear()
    out = tactical_insights(GAME_STATE, AWAY, 4, "FWD2")
    assert "Ball is on the FAR side" in out, out
    rec = out.split("Recovery position (goal-side of the ball): (")[1].split(")")[0]
    assert float(rec.split(",")[1]) == TUCK_IN_Y, out
    print(f"=== ALA: weak side tucks in to ({rec}) ===")


def test_ala_covers_wing_and_stays_out_of_box():
    rec, weak = ala_recovery({"x": -50, "y": 20}, -55.0, 1.0, DROP_LIMIT["FWD2"])
    assert not weak and rec["y"] > TUCK_IN_Y, rec
    assert rec["x"] >= -55 + DROP_LIMIT["FWD2"], rec
    print(f"=== ALA: ball side covers the wing at {rec} ===")


def test_ala_receives_wide():
    gs = _with_holder("home", 2, {"x": 5, "y": -8})
    out = tactical_insights(gs, HOME, 4, "FWD2")
    spot = out.split("Open space to receive: (")[1].split(")")[0]
    x, y = (float(v) for v in spot.split(","))
    assert abs(x - (55 - ALA_DEPTH)) <= 5 and y > 0, out
    print(f"=== ALA: receives wide right at ({x}, {y}) ===")


def test_counter_press_window():
    """Losing the ball close by opens the counter-press for a few seconds only."""
    _last_possession.clear()
    tactical_insights(GAME_STATE, HOME, 3, "FWD1")  # t=120.5, home holds it
    lost = copy.deepcopy(GAME_STATE)
    lost["ball"]["possessionAgentId"] = "agentId_1"
    lost["players"] = [p for p in lost["players"]
                       if not (p["teamCode"] == "home" and p["agentId"] == "agentId_1")]
    lost["gameTime"] = 122.0
    assert "COUNTER-PRESS now" in tactical_insights(lost, HOME, 3, "FWD1")
    lost["gameTime"] = 130.0
    assert "COUNTER-PRESS" not in tactical_insights(lost, HOME, 3, "FWD1")
    print("=== COUNTER-PRESS: open at +1.5s, closed at +9.5s ===")


def test_recovery_position_stays_out_of_own_box():
    from tactics import recovery_position
    home = recovery_position({"x": -50, "y": 10}, -55.0)
    away = recovery_position({"x": 50, "y": 10}, 55.0)
    assert home["x"] >= -40 and away["x"] <= 40, (home, away)
    print(f"=== RECOVERY: clamped outside the box home={home} away={away} ===")


class _FakeAgent:
    def __init__(self, reply: str):
        self.reply = reply
        self.messages = []
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        self.messages.extend([{"role": "user"}, {"role": "assistant"}])
        return self.reply


def _invoke(handler, game_state, team_id=HOME):
    payload = {"prompt": json.dumps({"teamId": team_id, "gameState": game_state})}

    async def collect():
        return [json.loads(chunk) async for chunk in handler(payload, None)]

    return asyncio.run(collect())[0]


def _handler(agent, **options):
    return create_invoke_handler(_FakeApp(), agent, 3, "FWD1", build_fallback(FWD1_CONFIG),
                                 fallback_cfg=FWD1_CONFIG, **options)


def test_handler_fast_path_skips_llm():
    agent = _FakeAgent("[]")
    handler = _handler(agent, fast_path_fn=fast_path)
    cmds = _invoke(handler, _with_holder("home", 3, {"x": 45, "y": 2}))
    assert cmds[0]["commandType"] == "SHOOT"
    assert agent.prompts == [], "LLM should not be called on a fast-path tick"
    print("=== HANDLER: fast path answered without calling the LLM ===")


def test_handler_stateless_clears_history_and_adds_context():
    reply = '[{"commandType":"PASS","parameters":{"target_player_id":4,"type":"GROUND"}}]'
    agent = _FakeAgent(reply)
    handler = _handler(agent, context_fn=tactical_insights, stateless=True)
    for _ in range(3):
        cmds = _invoke(handler, GAME_STATE)
    assert cmds[0]["commandType"] == "PASS"
    assert len(agent.messages) == 2, f"history grew to {len(agent.messages)}"
    assert "Tactical analysis" in agent.prompts[-1]
    print("=== HANDLER: history reset each tick, insights in prompt ===")


def test_handler_defaults_unchanged():
    """Teams that do not opt in keep their history (the memory team relies on it)."""
    agent = _FakeAgent('[{"commandType":"SET_STANCE","parameters":{"stance":1}}]')
    handler = _handler(agent)
    for _ in range(3):
        _invoke(handler, GAME_STATE)
    assert len(agent.messages) == 6
    assert "Tactical analysis" not in agent.prompts[-1]
    print("=== HANDLER: default behaviour unchanged ===")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"{len(fns)} tests passed")
