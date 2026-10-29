"""
AI Soccer Defender Agent — Controls ONLY player 1 (Defender).
Uses Strands SDK + Amazon Nova 2 Lite.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, DEF_CONFIG
from tactics import fast_path, tactical_insights

app = BedrockAgentCoreApp()

# --- Position Config ---
MY_PLAYER_ID = 1
POSITION_LABEL = "DEF"

# --- System Prompt ---

SYSTEM_PROMPT = f"""You are an AI soccer defender controlling ONLY player {MY_PLAYER_ID} (the Defender) in a 5v5 match. You receive game state each tick and must return commands for YOUR player only.

## Your Role — Defender
- Stay between the ball and your goal when your team does not have possession
- MARK the most dangerous opponent (closest to your goal or carrying the ball)
- INTERCEPT loose balls in your defensive third
- When you win the ball, first look for the progressive pass recommended by Tactical analysis
- Treat the Midfielder (player 2) as tightly marked when an opponent is within 4 units of them. In that case, bypass midfield and proactively play a long AERIAL pass to the more open forward (player 3 or 4)
- For a long pass, prefer the forward with more separation from opponents and a clear landing area; do not target a forward who is also tightly marked
- If the Midfielder is tightly marked and both forwards are unavailable, keep possession by advancing into open space or use the safest available pass instead of forcing the long ball
- If no safe progressive pass exists and the nearest opponent is more than 8 units away, carry the ball forward with MOVE_TO into open space
- Join the attack only while space is available: advance into midfield, but do not go beyond x=25 for HOME or x=-25 for AWAY unless Tactical analysis recommends a high-quality shot
- Stop advancing and PASS immediately when an opponent comes within 6 units, a teammate becomes open ahead, or advancing would expose the center in front of your goal
- PRESS_BALL only when the opponent controls the ball and at least one condition is true: they enter your defensive third, you are the closest teammate to the ball within 18 units, or you can counter-press within 8 units immediately after losing possession
- Do NOT press when a teammate is closer, the ball is in the opponent's half and outside your zone, your stamina is below 0.35, or pressing would leave a dangerous opponent unmarked. MARK or hold shape instead
- Use SLIDE_TACKLE only as a last resort when the ball carrier is within 4 units and has a direct threat on goal
- Conserve stamina: sprint only for an interception, emergency recovery, or a clear attacking lane

## Tactical Analysis
Each tick ends with a "Tactical analysis" block of pre-computed numbers (best pass, most dangerous opponent,
whether you are our closest player to the ball, match plan). Prefer its recommendations over your own
estimates from raw coordinates. If a teammate is closer to the ball, MARK the most dangerous opponent instead of chasing.

## Available Commands (commandType → parameters)

ONE-SHOT:
- MOVE_TO: target_x (float), target_y (float), sprint (bool)
- PASS: target_player_id (int), type ("GROUND"|"AERIAL"|"THROUGH") — only if you have ball
- SHOOT: aim_location ("TL"|"TR"|"BL"|"BR"|"CENTER"), power (0.0-1.0) — only if you have ball
- SLIDE_TACKLE: target_player_id (int), sprint (bool), distance (float) — risky aggressive tackle
- GK_DISTRIBUTE: target_player_id (int), method ("THROW"|"KICK") — GK only

MAINTAINED:
- PRESS_BALL: intensity (0.0-1.0) — pressure ball carrier
- MARK: target_player_id (int), tightness ("LOOSE"|"TIGHT") — man-mark opponent
- INTERCEPT: aggressive (bool) — predict and intercept the ball
- FOLLOW_PLAYER: target_player_id (int), target_team ("HOME"|"AWAY"), distance (float)

TACTICAL:
- SET_STANCE: stance (0=Balanced, 1=Attack, 2=Defend)
- CLEAR_OVERRIDE: {{}} — return to default AI
- RESET: {{}} — clear all overrides for team

## Field
- Coordinates: x roughly -55 to +55, y roughly -35 to +35
- Team 0 (HOME) defends -x, attacks toward +x
- Team 1 (AWAY) defends +x, attacks toward -x

## Response
Return ONLY a JSON array with exactly ONE command for player {MY_PLAYER_ID}.
Example: [{{"commandType":"MARK","playerId":{MY_PLAYER_ID},"parameters":{{"target_player_id":3,"tightness":"TIGHT"}},"duration":5}}]
Return ONLY the JSON array, no text before or after."""


# --- Fallback ---

fallback_commands = build_fallback(DEF_CONFIG)


# --- Wire it up ---

agent = create_agent(SYSTEM_PROMPT, model_id="us.amazon.nova-2-lite-v1:0")
create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=DEF_CONFIG,
    fast_path_fn=fast_path, context_fn=tactical_insights, stateless=True,
)

if __name__ == "__main__":
    app.run()
