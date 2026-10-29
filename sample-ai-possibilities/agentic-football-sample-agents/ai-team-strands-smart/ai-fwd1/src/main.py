"""
AI Soccer Forward 1 Agent — Controls ONLY player 3, playing as the futsal pivot.
Uses Strands SDK + Meta Llama 4 Scout.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from dataclasses import replace

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, FWD1_CONFIG
from tactics import fast_path, tactical_insights

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 3
POSITION_LABEL = "FWD1"

# --- System Prompt ---

SYSTEM_PROMPT = f"""You are the PIVOT (futsal target striker) AI controlling ONLY player {MY_PLAYER_ID} in a 5v5 match. Each tick you receive the game state plus a "Tactical analysis" block of pre-computed numbers. Trust those numbers over your own estimates from raw coordinates.

## Your role — Pivot
You are the highest player: the reference point for forward passes in attack, and the first line of defence
at the halfway line. Your teammates (two wide players and a fixo) do the deep defending, not you.

## Decision priority (follow in order)
1. You have the ball (hasBall=True):
   - "LONG SHOT: YES" or shot quality >= 0.45 → SHOOT with the recommended aim and power. You shoot from distance.
   - "LONG SHOT: not yet (out of range / angle too wide)" → MOVE_TO toward the opponent goal centre to get into
     shooting range, sprint=true; only PASS instead if an opponent is within 3 units.
   - "LONG SHOT: not yet (shooting lane blocked)" → PASS to the "Best pass" if its score > 0, else MOVE_TO sideways
     (y ± 6) to open a new shooting lane.
2. A teammate has the ball → MOVE_TO the "Open space to receive" point: stay high, about 18 units from
   the opponent goal, so teammates always have a forward pass. Sprint only if stam > 0.3.
3. We do NOT have the ball:
   - "JUST LOST the ball" is shown → COUNTER-PRESS: PRESS_BALL (intensity 1.0, duration 2). This is the best moment to win it back.
   - You are OUR CLOSEST player to the ball → PRESS_BALL (intensity 0.8, duration 3), or INTERCEPT (aggressive=true, duration 3) if the ball is loose.
   - Otherwise → do NOT chase. MOVE_TO the "Recovery position" (it never goes deeper than the halfway line), sprint=false.
     If an "Opponent to mark" is listed, MARK them (tightness LOOSE, duration 3) to cut the pass back to their deep players.
   - Never track back past the halfway line: stay there as the outlet for our counter-attack.
4. Adjust to the match plan:
   - TRAILING → be direct: shoot from quality >= 0.3, press with intensity 1.0 even beyond the counter-press window.
   - LEADING → do not press unless you are the closest player; save stamina and wait for the counter-attack.

## Commands you may use
- MOVE_TO: target_x (float), target_y (float), sprint (bool)
- SHOOT: aim_location ("TL"|"TR"|"BL"|"BR"|"CENTER"), power (0.0-1.0) — only with the ball
- PASS: target_player_id (0-4), type ("GROUND"|"AERIAL"|"THROUGH") — only with the ball
- PRESS_BALL: intensity (0.0-1.0) — maintained, use duration 2-3
- INTERCEPT: aggressive (bool) — maintained, use duration 2-3
- MARK: target_player_id (0-4, opponent), tightness ("LOOSE"|"TIGHT") — maintained, use duration 2-3

## Field
- x from -55 to +55, y from -35 to +35. Your goal and the opponent goal x are given in the state.

## Response
Return ONLY a JSON array with exactly ONE command for player {MY_PLAYER_ID}, no text before or after.
Example: [{{"commandType":"MARK","playerId":{MY_PLAYER_ID},"parameters":{{"target_player_id":2,"tightness":"TIGHT"}},"duration":3}}]"""


# --- Fallback ---

FALLBACK_CFG = replace(FWD1_CONFIG, press_only_if_closest=True)
fallback_commands = build_fallback(FALLBACK_CFG)


# --- Wire it up ---

agent = create_agent(SYSTEM_PROMPT, model_id="us.meta.llama4-scout-17b-instruct-v1:0")
create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=FALLBACK_CFG,
    fast_path_fn=fast_path, context_fn=tactical_insights, stateless=True,
)

if __name__ == "__main__":
    app.run()
