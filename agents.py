"""
agents.py
─────────
Three specialized async agents powered by Ollama (gemma3).

  Agent A — Farmer       : Soil micronutrients + weather analysis.
  Agent B — Trader: Mandi price trends + logistics.
  Agent C — Analyst : 2026 MSPs + export policies.

Phase 1: All three agents run in PARALLEL via asyncio.gather(), each
         analysing all Top-5 crops with a score (1–10) + reason.

Phase 2: CROSS-CRITIQUE LOOP — Farmer ↔ Trader exchange
         reasons and revise scores iteratively until consensus (Δ ≤ 1)
         or max rounds are reached.  Analyst scores are fixed.

WebSocket: A FastAPI server broadcasts real-time agent debate logs to
           all connected frontend clients via /ws.
"""

import asyncio
import json
import sys
import time
from typing import Any

from ollama import AsyncClient
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

from data_loader import (
    load_json,
    get_bundle,
    get_top5_crops,
    get_deficient_micronutrients,
    get_daily_forecasts,
    get_agro_advisory,
    get_mandi_prices,
    get_msp_data,
    get_export_signals,
    get_metadata,
    get_soil_health_card,
)


# ──────────────────────────────────────────────
#  WebSocket Connection Manager
# ──────────────────────────────────────────────

class ConnectionManager:
    """Tracks active WebSocket clients and broadcasts messages."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"  🔌 WebSocket client connected ({len(self.active_connections)} active)")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"  🔌 WebSocket client disconnected ({len(self.active_connections)} active)")

    async def broadcast(self, message: dict):
        """Send a JSON message to ALL connected clients."""
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.active_connections.remove(conn)


manager = ConnectionManager()


async def broadcast_log(
    sender: str,
    text: str,
    *,
    crop: str = "",
    phase: str = "",
    event: str = "agent_log",
    score: int | None = None,
) -> None:
    """Broadcast a structured log entry to all connected WebSocket clients.

    This is fire-and-forget — it never blocks the agent pipeline.

    Args:
        sender: Agent name (e.g. 'Farmer', 'System').
        text: The reasoning / status text.
        crop: Optional crop name this log pertains to.
        phase: Pipeline phase label (e.g. 'phase_1', 'critique_r2').
        event: Message type for the frontend to route on.
        score: Optional numeric score.
    """
    msg = {
        "event": event,
        "sender": sender,
        "crop": crop,
        "text": text,
        "phase": phase,
        "timestamp": time.time(),
    }
    if score is not None:
        msg["score"] = score
    await manager.broadcast(msg)

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

MODEL = "gemma3"

# JSON schema that every agent MUST follow
RESPONSE_SCHEMA = """
{
  "agent_name": "<string>",
  "analysis": [
    {
      "crop_name": "<string>",
      "score": <integer 1-10>,
      "reason": "<one sentence>"
    }
  ]
}
""".strip()


# ──────────────────────────────────────────────
#  Prompt Builders
# ──────────────────────────────────────────────

def _build_agronomist_prompt(bundle: dict) -> str:
    """Build the system + user prompt for Agent A (Farmer).

    Focus: soil micronutrients + weather.
    Rule : If Zinc or Boron are low/deficient OR a dry spell is
           predicted, PENALIZE high-water-requirement crops.
    """
    top5 = get_top5_crops(bundle)
    deficiencies = get_deficient_micronutrients(bundle)
    forecasts = get_daily_forecasts(bundle)
    advisory = get_agro_advisory(bundle)
    meta = get_metadata(bundle)

    crop_list = json.dumps(
        [
            {
                "crop_name": c["crop_name"],
                "water_requirement_mm": c["water_requirement_mm"],
                "growing_period_days": c["growing_period_days"],
                "suitability_score": c["suitability_score"],
            }
            for c in top5
        ],
        indent=2,
    )

    deficiency_text = json.dumps(deficiencies, indent=2)
    forecast_text = json.dumps(forecasts, indent=2)
    advisory_text = json.dumps(advisory, indent=2)

    return f"""You are **Agent A — Farmer**, an expert in soil science and agro-meteorology.

## Your Region
- Region: {meta.get('region', 'N/A')}, District: {meta.get('district', 'N/A')}
- Season: {meta.get('season', 'N/A')}
- Soil Type: {meta.get('soil_type', 'N/A')}
- Irrigation: {meta.get('irrigation_source', 'N/A')}

## Soil Micronutrient Deficiencies (from Soil Health Card)
{deficiency_text}

## 2-Day Weather Forecast
{forecast_text}

## Agro-Meteorological Advisory
{advisory_text}

## Top 5 Candidate Crops
{crop_list}

## Your Task
Analyze ALL 5 crops above. For each crop, assign a score from 1 (worst) to 10 (best) based on:
1. Compatibility with the current soil micronutrient deficiencies (Zinc, Copper, Boron are deficient/low).
2. Weather suitability over the next 2 days and upcoming season.
3. **CRITICAL RULE**: If Zinc OR Boron levels are low/deficient, AND a dry spell (low rainfall < 5mm with high temperatures > 37°C) is predicted, you MUST PENALIZE crops with high water requirements (> 1000mm). Reduce their score by at least 2-3 points.

Provide exactly one short sentence as the reason for each score.

## Output Format (strict JSON, no extra text)
{RESPONSE_SCHEMA}

Set "agent_name" to "Farmer".
Respond ONLY with the JSON object. No markdown fences, no commentary."""


def _build_market_strategist_prompt(bundle: dict) -> str:
    """Build the system + user prompt for Agent B (Trader).

    Focus: Mandi price trends + logistics.
    Rule : Upvote crops whose mandi prices are rising/spiking.
    """
    top5 = get_top5_crops(bundle)
    mandi = get_mandi_prices(bundle)
    msp = get_msp_data(bundle)

    crop_names = [c["crop_name"] for c in top5]

    crop_list = json.dumps(
        [
            {
                "crop_name": c["crop_name"],
                "predicted_yield_tons_per_ha": c["predicted_yield_tons_per_ha"],
                "growing_period_days": c["growing_period_days"],
            }
            for c in top5
        ],
        indent=2,
    )

    mandi_text = json.dumps(mandi, indent=2)
    msp_text = json.dumps(msp, indent=2)

    return f"""You are **Agent B — Trader**, an expert in agricultural commodity markets and supply-chain logistics.

## Current Mandi (Market) Prices — Thanjavur APMC Snapshot
{mandi_text}

## Minimum Support Prices (MSP) — 2026 Kharif Season
{msp_text}

## Top 5 Candidate Crops
{crop_list}

## Your Task
Analyze ALL 5 crops above. For each crop, assign a score from 1 (worst) to 10 (best) based on:
1. Current mandi price trends — **UPVOTE crops whose prices are "rising" or "spiking"**. Rising-trend crops should get a +2 to +3 bonus.
2. Gap between current mandi modal price and MSP — a mandi price above MSP signals strong demand.
3. Logistics and marketability — shorter growing periods mean faster cash flow.
4. Price stability — "stable" is neutral; "declining" should be penalized.

Provide exactly one short sentence as the reason for each score.

## Output Format (strict JSON, no extra text)
{RESPONSE_SCHEMA}

Set "agent_name" to "Trader".
Respond ONLY with the JSON object. No markdown fences, no commentary."""


def _build_trend_forecaster_prompt(bundle: dict) -> str:
    """Build the system + user prompt for Agent C (Analyst).

    Focus: 2026 MSPs + export policies.
    """
    top5 = get_top5_crops(bundle)
    msp = get_msp_data(bundle)
    exports = get_export_signals(bundle)

    crop_list = json.dumps(
        [
            {
                "crop_name": c["crop_name"],
                "suitability_score": c["suitability_score"],
                "predicted_yield_tons_per_ha": c["predicted_yield_tons_per_ha"],
            }
            for c in top5
        ],
        indent=2,
    )

    msp_text = json.dumps(msp, indent=2)
    export_text = json.dumps(exports, indent=2)

    return f"""You are **Agent C — Analyst**, an expert in agricultural policy analysis, government support mechanisms, and global trade patterns.

## Minimum Support Prices (MSP) — 2026 Kharif Season
{msp_text}

## Export Demand Signals — 2026
{export_text}

## Top 5 Candidate Crops
{crop_list}

## Your Task
Analyze ALL 5 crops above. For each crop, assign a score from 1 (worst) to 10 (best) based on:
1. **MSP year-on-year change** — crops with higher MSP increases (> 5%) signal stronger government support; reward them.
2. **Export demand** — "Strong" export demand deserves a significant bonus (+2–3); "Moderate" is neutral; export restrictions or domestic-only focus should be penalized.
3. **Policy tailwinds** — consider ethanol blending mandates (benefits sugarcane/maize), food security policies (benefits paddy), and pulse mission support (benefits black gram).
4. **Long-term trend viability** — crops aligned with India's 2026 agri-export targets score higher.

Provide exactly one short sentence as the reason for each score.

## Output Format (strict JSON, no extra text)
{RESPONSE_SCHEMA}

Set "agent_name" to "Analyst".
Respond ONLY with the JSON object. No markdown fences, no commentary."""


# ──────────────────────────────────────────────
#  Agent Runner (generic async Ollama call)
# ──────────────────────────────────────────────

async def _call_agent(agent_label: str, prompt: str, *, phase: str = "") -> dict:
    """Send the prompt to Ollama (gemma3) and parse the JSON response.

    Args:
        agent_label: Human-readable name (for logging).
        prompt: The full prompt string.
        phase: Pipeline phase label (e.g. 'phase_1', 'critique_r2').

    Returns:
        Parsed dict from the model's JSON response.
    """
    client = AsyncClient()
    print(f"  🚀 [{agent_label}] Sending request to Ollama ({MODEL})…")

    dispatch_text = "Analyzing data..."
    status_text = "Here are my scores:"
    
    if "critique" in phase:
        if "Farmer" in agent_label:
            dispatch_text = "Let me review the Trader's points. I might need to adjust my scores based on their market view..."
            status_text = "I've re-evaluated the crops taking the market into account. Here are my revised thoughts:"
        else:
            dispatch_text = "Reviewing the Farmer's agronomic concerns. Let's see if this changes my market strategy..."
            status_text = "Done. Here is my revised outlook:"
    else:
        if "Farmer" in agent_label:
            dispatch_text = "I'm looking at the soil report and the upcoming weather. Give me a moment to analyze how the crops will fare..."
        elif "Trader" in agent_label:
            dispatch_text = "Checking the current mandi prices and the 2026 MSP data... I'll let you know my thoughts shortly."
        elif "Analyst" in agent_label:
            dispatch_text = "I'm reviewing the export policies and global demand trends. Crunching the numbers now..."
        else:
            dispatch_text = "Analyzing data..."
        status_text = "Alright, here are my initial scores for the Top 5 crops:"

    # Broadcast: agent is starting
    await broadcast_log(
        sender=agent_label,
        text=dispatch_text,
        phase=phase,
        event="agent_dispatch",
    )

    t0 = time.perf_counter()

    response = await client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3},          # Low temp for deterministic output
        format="json",                          # Request JSON mode
    )

    elapsed = time.perf_counter() - t0
    raw = response["message"]["content"]
    print(f"  ✅ [{agent_label}] Response received in {elapsed:.1f}s")

    # Broadcast: agent finished
    await broadcast_log(
        sender=agent_label,
        text=status_text,
        phase=phase,
        event="agent_status",
    )

    # Parse the JSON; strip markdown fences if model wraps them
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0]

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  [{agent_label}] JSON parse failed, returning raw text.")
        result = {"agent_name": agent_label, "raw_response": raw, "parse_error": str(e)}

    # Broadcast: each crop's score + reasoning immediately
    for entry in result.get("analysis", []):
        await broadcast_log(
            sender=agent_label,
            crop=entry.get("crop_name", ""),
            text=entry.get("reason", ""),
            score=entry.get("score"),
            phase=phase,
        )

    return result


# ──────────────────────────────────────────────
#  Individual Agent Entry Points
# ──────────────────────────────────────────────

async def run_farmer(bundle: dict, *, phase: str = "phase_1") -> dict:
    """Agent A — Farmer: Soil micronutrients + weather."""
    prompt = _build_agronomist_prompt(bundle)
    return await _call_agent("Farmer", prompt, phase=phase)


async def run_trader(bundle: dict, *, phase: str = "phase_1") -> dict:
    """Agent B — Trader: Mandi price trends + logistics."""
    prompt = _build_market_strategist_prompt(bundle)
    return await _call_agent("Trader", prompt, phase=phase)


async def run_analyst(bundle: dict, *, phase: str = "phase_1") -> dict:
    """Agent C — Analyst: 2026 MSPs + export policies."""
    prompt = _build_trend_forecaster_prompt(bundle)
    return await _call_agent("Analyst", prompt, phase=phase)


# ──────────────────────────────────────────────
#  Parallel Orchestrator
# ──────────────────────────────────────────────

async def run_all_agents(bundle: dict) -> list[dict]:
    """Execute all three agents in PARALLEL and return their results.

    Uses asyncio.gather() so Agent A, B, and C run concurrently
    rather than sequentially.
    """
    print("\n⏳ Dispatching all 3 agents in parallel…\n")

    await broadcast_log(
        sender="System",
        text="Phase 1: Dispatching Farmer, Trader, and Analyst in parallel",
        phase="phase_1",
        event="phase_start",
    )

    t0 = time.perf_counter()

    results = await asyncio.gather(
        run_farmer(bundle, phase="phase_1"),
        run_trader(bundle, phase="phase_1"),
        run_analyst(bundle, phase="phase_1"),
    )

    elapsed = time.perf_counter() - t0
    print(f"\n🏁 All agents completed in {elapsed:.1f}s (parallel)\n")

    await broadcast_log(
        sender="System",
        text=f"Phase 1 complete — all 3 agents finished in {elapsed:.1f}s",
        phase="phase_1",
        event="phase_end",
    )

    return list(results)


# ──────────────────────────────────────────────
#  Pretty-Print Results
# ──────────────────────────────────────────────

def display_results(results: list[dict]) -> None:
    """Print a formatted summary of all agent outputs."""
    print("═" * 70)
    print("  🤖  MULTI-AGENT ANALYSIS RESULTS")
    print("═" * 70)

    for res in results:
        agent_name = res.get("agent_name", "Unknown Agent")
        analysis = res.get("analysis", [])

        print(f"\n┌{'─' * 68}┐")
        print(f"│  🧠 {agent_name:<63}│")
        print(f"├{'─' * 68}┤")

        if analysis:
            for entry in analysis:
                crop = entry.get("crop_name", "?")
                score = entry.get("score", "?")
                reason = entry.get("reason", "No reason provided.")
                bar = "█" * int(score) if isinstance(score, (int, float)) else ""
                print(f"│  {crop:<28} Score: {score:<3} {bar:<10} │")
                print(f"│    → {reason:<62}│")
        else:
            # Fallback for raw / unparsed responses
            raw = res.get("raw_response", json.dumps(res, indent=2))
            for line in raw.split("\n")[:6]:
                print(f"│  {line:<66}│")

        print(f"└{'─' * 68}┘")

    print()


# ──────────────────────────────────────────────
#  Phase 2: Cross-Critique Helpers
# ──────────────────────────────────────────────

MAX_CRITIQUE_ROUNDS = 3
CONSENSUS_THRESHOLD = 1  # max score delta to declare convergence

# Canonical crop names (must match all_agent_data.json top_5_crops)
CANONICAL_CROPS = [
    "Paddy (IR-64)",
    "Black Gram (Urad Dal)",
    "Sugarcane (Co-86032)",
    "Groundnut (TMV-7)",
    "Maize (Hybrid NK-6240)",
]


def _fuzzy_match_crop(name: str) -> str:
    """Match a possibly-mangled crop name to the canonical list."""
    low = name.lower()
    for canon in CANONICAL_CROPS:
        # Match if the base name (before parenthesis) appears in the response
        base = canon.split(" (")[0].lower()
        if base in low or low in canon.lower():
            return canon
    return name  # fallback: use as-is


def _extract_analysis_map(result: dict) -> dict[str, dict]:
    """Convert an agent result into {canonical_crop_name: {score, reason}}."""
    out = {}
    for e in result.get("analysis", []):
        key = _fuzzy_match_crop(e.get("crop_name", ""))
        out[key] = {"score": e.get("score", 0), "reason": e.get("reason", "N/A")}
    return out


def _format_peer_feedback(peer_name: str, peer_result: dict) -> str:
    """Format another agent's analysis as readable text for a prompt."""
    lines = [f"### Feedback from {peer_name}"]
    for entry in peer_result.get("analysis", []):
        lines.append(
            f"- **{entry['crop_name']}** → Score {entry['score']}/10: "
            f"\"{entry['reason']}\""
        )
    return "\n".join(lines)


def _check_consensus(prev: dict, curr: dict) -> tuple[bool, float]:
    """Check if two rounds' scores have converged.

    Returns (converged: bool, max_delta: float).
    """
    if not prev or not curr:
        return False, 99.0
    max_delta = 0.0
    for crop in curr:
        p = prev.get(crop, {}).get("score", 0)
        c = curr[crop].get("score", 0)
        max_delta = max(max_delta, abs(p - c))
    return max_delta <= CONSENSUS_THRESHOLD, max_delta


def _build_critique_prompt(
    agent_name: str,
    own_domain: str,
    own_result: dict,
    peer_name: str,
    peer_result: dict,
    bundle: dict,
    round_num: int,
) -> str:
    """Build a cross-critique prompt for one agent."""
    meta = get_metadata(bundle)
    own_feedback = _format_peer_feedback(f"Your previous analysis ({agent_name})", own_result)
    peer_feedback = _format_peer_feedback(peer_name, peer_result)

    soil_context = ""
    if agent_name == "Farmer":
        deficiencies = get_deficient_micronutrients(bundle)
        forecasts = get_daily_forecasts(bundle)
        soil_context = f"""
## Your Domain Data (Soil & Weather)
- Micronutrient deficiencies: {json.dumps(deficiencies)}
- 2-day forecast: {json.dumps(forecasts)}
- Soil pH: 7.2, OC: 0.58%, N: 285 kg/ha (Medium), P: 22.5 kg/ha (Medium), K: 310 kg/ha (High)
"""

    market_context = ""
    if agent_name == "Trader":
        mandi = get_mandi_prices(bundle)
        msp = get_msp_data(bundle)
        market_context = f"""
## Your Domain Data (Market)
- Mandi prices: {json.dumps(mandi)}
- MSP data: {json.dumps(msp)}
"""

    return f"""You are **{agent_name}** (specialist in {own_domain}).
This is Cross-Critique Round {round_num}.

Region: {meta.get('district', 'N/A')}, {meta.get('region', 'N/A')} | Season: {meta.get('season', 'N/A')}
{soil_context}{market_context}
{own_feedback}

{peer_feedback}

## Your Task
1. Read the {peer_name}'s scores and reasons carefully.
2. Cross-check their claims against YOUR domain expertise:
   - If {peer_name} rates a crop highly but your data shows a problem (e.g., soil deficiency, weather risk, declining price), you should maintain or lower your score and explain why.
   - If {peer_name} raises a valid concern you missed, adjust your score accordingly.
3. For EACH of the 5 crops, provide your REVISED score (1-10) and a one-sentence reason that acknowledges the cross-critique.

## Output Format (strict JSON, no extra text)
{RESPONSE_SCHEMA}

Set "agent_name" to "{agent_name}".
Respond ONLY with the JSON object."""


# ──────────────────────────────────────────────
#  Cross-Critique Loop
# ──────────────────────────────────────────────

async def run_cross_critique(
    bundle: dict,
    farmer_result: dict,
    trader_result: dict,
) -> tuple[dict, dict, int]:
    """Run the cross-critique loop between Farmer and Trader.

    Each round:
      1. Farmer receives Trader's reasons → re-scores (parallel)
      2. Trader receives Farmer's reasons → re-scores (parallel)
      3. Check convergence on BOTH agents.

    Returns:
        (final_farmer, final_trader, rounds_taken)
    """
    prev_farmer_map: dict = {}
    prev_market_map: dict = {}
    current_farmer = farmer_result
    current_trader = trader_result

    for rnd in range(1, MAX_CRITIQUE_ROUNDS + 1):
        critique_phase = f"critique_r{rnd}"

        print(f"\n{'─' * 60}")
        print(f"  🔄  CROSS-CRITIQUE ROUND {rnd}/{MAX_CRITIQUE_ROUNDS}")
        print(f"{'─' * 60}\n")

        # Broadcast: round start
        await broadcast_log(
            sender="System",
            text=f"Cross-Critique Round {rnd}/{MAX_CRITIQUE_ROUNDS} — agents exchanging feedback",
            phase=critique_phase,
            event="critique_round_start",
        )

        agro_prompt = _build_critique_prompt(
            "Farmer", "soil micronutrients & weather",
            current_farmer, "Trader", current_trader,
            bundle, rnd,
        )
        market_prompt = _build_critique_prompt(
            "Trader", "mandi prices & logistics",
            current_trader, "Farmer", current_farmer,
            bundle, rnd,
        )

        # Both agents critique in PARALLEL
        new_farmer, new_trader = await asyncio.gather(
            _call_agent(f"Farmer  (critique R{rnd})", agro_prompt, phase=critique_phase),
            _call_agent(f"Trader (critique R{rnd})", market_prompt, phase=critique_phase),
        )

        # Check convergence
        new_farmer_map = _extract_analysis_map(new_farmer)
        new_trader_map = _extract_analysis_map(new_trader)
        old_farmer_map = _extract_analysis_map(current_farmer)
        old_trader_map = _extract_analysis_map(current_trader)

        agro_conv, agro_delta = _check_consensus(old_farmer_map, new_farmer_map)
        mkt_conv, mkt_delta = _check_consensus(old_trader_map, new_trader_map)

        agro_status = "converged" if agro_conv else "shifting"
        mkt_status = "converged" if mkt_conv else "shifting"

        print(f"\n  📊 Farmer  Δmax={agro_delta:.0f}  {'✅ converged' if agro_conv else '🔸 shifting'}")
        print(f"  📊 Trader   Δmax={mkt_delta:.0f}  {'✅ converged' if mkt_conv else '🔸 shifting'}")

        # Broadcast: convergence status for each agent
        await broadcast_log(
            sender="Farmer",
            text="I'm satisfied with my scores now." if agro_conv else "I need to make a few more adjustments.",
            phase=critique_phase,
            event="convergence_check",
        )
        await broadcast_log(
            sender="Trader",
            text="My market outlook is locked in." if mkt_conv else "I'm still adjusting my scores.",
            phase=critique_phase,
            event="convergence_check",
        )

        current_farmer = new_farmer
        current_trader = new_trader

        if agro_conv and mkt_conv:
            print(f"\n  🤝 CONSENSUS reached after {rnd} round(s)!")
            await broadcast_log(
                sender="System",
                text=f"🤝 CONSENSUS reached after {rnd} round(s)!",
                phase=critique_phase,
                event="consensus_reached",
            )
            return current_farmer, current_trader, rnd

    print(f"\n  ⏰ Max rounds ({MAX_CRITIQUE_ROUNDS}) reached — using latest scores.")
    await broadcast_log(
        sender="System",
        text=f"Max rounds ({MAX_CRITIQUE_ROUNDS}) reached — using latest scores",
        phase=f"critique_r{MAX_CRITIQUE_ROUNDS}",
        event="critique_max_rounds",
    )
    return current_farmer, current_trader, MAX_CRITIQUE_ROUNDS


def display_consensus(agro: dict, market: dict, trend: dict, rounds: int) -> None:
    """Print a final merged consensus table."""
    farmer_map = _extract_analysis_map(agro)
    trader_map = _extract_analysis_map(market)
    analyst_map = _extract_analysis_map(trend)

    print("\n" + "═" * 70)
    print(f"  🤝  FINAL CONSENSUS (after {rounds} critique round(s))")
    print("═" * 70)
    print(f"\n  {'Crop':<28}{'Agro':>6}{'Market':>8}{'Trend':>7}{'  Avg':>6}")
    print(f"  {'─'*26:<28}{'─'*4:>6}{'─'*5:>8}{'─'*5:>7}{'─'*5:>6}")

    # Use canonical crop names as the master key list
    all_crops = CANONICAL_CROPS

    rows = []
    for crop_name in all_crops:
        a = farmer_map.get(crop_name, {}).get("score", 0)
        m = trader_map.get(crop_name, {}).get("score", 0)
        t = analyst_map.get(crop_name, {}).get("score", 0)
        avg = round((a + m + t) / 3, 1)
        rows.append((crop_name, a, m, t, avg))
    rows.sort(key=lambda r: r[4], reverse=True)

    for i, (crop_name, a, m, t, avg) in enumerate(rows):
        medal = "🥇" if i == 0 else "  "
        print(f"  {medal}{crop_name:<26}{a:>6}{m:>8}{t:>7}{avg:>6}")

    # Print revised reasons
    print(f"\n  {'─' * 68}")
    print("  📝 REVISED REASONING")
    print(f"  {'─' * 68}")
    for crop_name in all_crops:
        ar = farmer_map.get(crop_name, {}).get("reason", "(no agronomist feedback)")
        mr = trader_map.get(crop_name, {}).get("reason", "(no market feedback)")
        print(f"\n  🌱 {crop_name}")
        print(f"     Farmer : {ar}")
        print(f"     Trader  : {mr}")

    print(f"\n{'═' * 70}\n")


# ──────────────────────────────────────────────
#  Phase 3: Final Viability Index (FVI)
# ──────────────────────────────────────────────
#
#  FVI = (ML_suitability × 0.4) + (S_agronomy × 0.2)
#        + (S_market × 0.2) + (S_demand × 0.2)
#
#  ML suitability is 0.0–1.0 in the JSON → scaled to 0–10
#  Agent scores are already 1–10
# ──────────────────────────────────────────────

def compute_fvi(
    bundle: dict,
    farmer_result: dict,
    trader_result: dict,
    analyst_result: dict,
) -> list[dict]:
    """Calculate the Final Viability Index for each Top-5 crop.

    Formula (all scores normalized to 0–10 scale):
        FVI = (ML_suitability × 0.4)
            + (S_agronomy     × 0.2)
            + (S_market        × 0.2)
            + (S_demand        × 0.2)

    Args:
        bundle: The knowledge_bundle dict (contains ML scores).
        farmer_result: Final Farmer agent output.
        trader_result: Final Trader agent output.
        analyst_result: Analyst agent output.

    Returns:
        List of dicts sorted by FVI descending. Each dict:
            rank, crop_name, ml_score, s_farmer, s_trader,
            s_analyst, fvi
    """
    top5 = get_top5_crops(bundle)
    farmer_map = _extract_analysis_map(farmer_result)
    trader_map = _extract_analysis_map(trader_result)
    analyst_map = _extract_analysis_map(analyst_result)

    results = []
    for crop in top5:
        name = crop["crop_name"]

        # ML suitability: 0.0–1.0 → scale to 0–10
        ml_raw = crop["suitability_score"]       # e.g. 0.94
        ml_scaled = ml_raw * 10                   # e.g. 9.4

        # Agent scores (already 1–10)
        s_farmer = farmer_map.get(name, {}).get("score", 0)
        s_trader   = trader_map.get(name, {}).get("score", 0)
        s_analyst   = analyst_map.get(name, {}).get("score", 0)

        # FVI formula
        fvi = (
            (ml_scaled  * 0.4)
          + (s_farmer * 0.2)
          + (s_trader   * 0.2)
          + (s_analyst   * 0.2)
        )

        results.append({
            "crop_name":  name,
            "ml_score":   round(ml_scaled, 2),
            "s_farmer": s_farmer,
            "s_trader":   s_trader,
            "s_analyst":   s_analyst,
            "fvi":        round(fvi, 2),
        })

    # Sort by FVI descending
    results.sort(key=lambda r: r["fvi"], reverse=True)

    # Assign final ranks
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results


def display_fvi(fvi_results: list[dict]) -> None:
    """Print a formatted FVI ranking table."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}

    print("\n" + "═" * 78)
    print("  📊  FINAL VIABILITY INDEX (FVI) RANKING")
    print("  FVI = (ML×0.4) + (Agro×0.2) + (Market×0.2) + (Demand×0.2)")
    print("═" * 78)
    print(
        f"\n  {'Rank':<6}{'Crop':<28}"
        f"{'ML':>5}{'Agro':>6}{'Mkt':>6}{'Dmd':>6}"
        f"{'FVI':>8}"
    )
    print(
        f"  {'─'*4:<6}{'─'*26:<28}"
        f"{'─'*4:>5}{'─'*4:>6}{'─'*4:>6}{'─'*4:>6}"
        f"{'─'*6:>8}"
    )

    for r in fvi_results:
        medal = medals.get(r["rank"], "  ")
        bar_len = int(r["fvi"])
        bar = "█" * bar_len + "░" * (10 - bar_len)
        print(
            f"  {medal}{r['rank']:<4}{r['crop_name']:<28}"
            f"{r['ml_score']:>5.1f}{r['s_farmer']:>6}"
            f"{r['s_trader']:>6}{r['s_analyst']:>6}"
            f"{r['fvi']:>8.2f}"
        )
        print(f"        {bar}")

    # Recommendation
    best = fvi_results[0]
    print(f"\n  {'─' * 76}")
    print(f"  ✅ RECOMMENDED CROP: {best['crop_name']}  (FVI = {best['fvi']:.2f})")
    print(f"  {'─' * 76}")
    print(f"\n{'═' * 78}\n")


# ──────────────────────────────────────────────
#  Phase 4: Intelligence Output
# ──────────────────────────────────────────────

def _build_fertilizer_advice(bundle: dict) -> list[dict]:
    """Pure Python: extract actionable fertilizer advice from SHC."""
    deficiencies = get_deficient_micronutrients(bundle)
    shc = get_soil_health_card(bundle)
    macro = shc.get("macronutrients", {})

    advice = []
    for d in deficiencies:
        advice.append({
            "nutrient": d["name"],
            "status": f"{d['value_ppm']} ppm ({d['rating']})",
            "action": d["recommendation"],
        })

    # Macro-nutrient notes for borderline values
    n = macro.get("nitrogen_kg_per_ha", {})
    if n.get("rating") == "Medium":
        advice.append({
            "nutrient": "Nitrogen",
            "status": f"{n['value']} kg/ha (Medium — lower end)",
            "action": "Apply Urea @ 60-80 kg/ha in 2 splits for cereal crops; reduce for pulses (natural N-fixation).",
        })

    return advice


def _build_sowing_advisory(bundle: dict) -> dict:
    """Pure Python: derive sowing timing from weather forecast."""
    forecasts = get_daily_forecasts(bundle)
    agro = get_agro_advisory(bundle)
    meta = get_metadata(bundle)

    rain_day1 = forecasts[0]["precipitation"]["expected_mm"] if forecasts else 0
    rain_day2 = forecasts[1]["precipitation"]["expected_mm"] if len(forecasts) > 1 else 0
    total_rain = rain_day1 + rain_day2

    temp_max = max(f["temperature"]["max_celsius"] for f in forecasts) if forecasts else 0

    window = agro.get("sowing_window_status", "N/A")
    pest = agro.get("pest_alert", "N/A")
    irrig = agro.get("irrigation_advisory", "N/A")

    # Derive timing recommendations per crop type
    timing = {}
    if total_rain >= 10:
        timing["pulses"] = "Sow immediately — adequate soil moisture from expected rainfall."
        timing["paddy"]  = "Wait for sustained rainfall (>50mm over 3 days) for transplanting."
    else:
        timing["pulses"] = "Dry-sow with pre-irrigation; monitor for germination moisture."
        timing["paddy"]  = "Defer transplanting until monsoon onset ensures standing water."

    if temp_max > 37:
        timing["sugarcane"] = "Avoid planting in current heat wave; schedule for early June."
    else:
        timing["sugarcane"] = "Conditions acceptable for sett planting with drip irrigation."

    timing["oilseeds_cereals"] = (
        "Sow after first significant rain (>15mm); "
        "current forecast shows rain tomorrow — prepare fields now."
        if rain_day2 >= 10 else
        "Await adequate moisture; prepare seed and field in the meantime."
    )

    return {
        "window_status": window,
        "pest_alert": pest,
        "irrigation_advisory": irrig,
        "expected_rain_48h_mm": total_rain,
        "peak_temp_celsius": temp_max,
        "crop_timing": timing,
    }


async def _generate_why_summaries(
    fvi_top3: list[dict],
    farmer_result: dict,
    trader_result: dict,
    analyst_result: dict,
) -> list[dict]:
    """One final Ollama call: synthesize a consolidated 'Why' for each top-3 crop."""
    farmer_map = _extract_analysis_map(farmer_result)
    trader_map = _extract_analysis_map(trader_result)
    analyst_map = _extract_analysis_map(analyst_result)

    crop_details = []
    for r in fvi_top3:
        name = r["crop_name"]
        crop_details.append({
            "rank": r["rank"],
            "crop_name": name,
            "fvi": r["fvi"],
            "ml_score": r["ml_score"],
            "farmer_reason": farmer_map.get(name, {}).get("reason", "N/A"),
            "trader_reason": trader_map.get(name, {}).get("reason", "N/A"),
            "analyst_reason": analyst_map.get(name, {}).get("reason", "N/A"),
        })

    prompt = f"""You are an agricultural intelligence analyst. Summarize why each of the top 3 crops
was ranked in its position. For EACH crop, write exactly ONE concise sentence (max 30 words)
that captures the key trade-off or advantage (e.g., "Soybean moved to #1 due to strong export
demand outweighing moderate water risk").

Crop data:
{json.dumps(crop_details, indent=2)}

Output strict JSON:
{{
  "summaries": [
    {{"rank": 1, "crop_name": "...", "why": "..."}},
    {{"rank": 2, "crop_name": "...", "why": "..."}},
    {{"rank": 3, "crop_name": "...", "why": "..."}}
  ]
}}
Respond ONLY with JSON."""

    result = await _call_agent("Intelligence Analyst", prompt)
    return result.get("summaries", [])


def display_intelligence_output(
    fvi_top3: list[dict],
    why_summaries: list[dict],
    fertilizer: list[dict],
    sowing: dict,
) -> None:
    """Print the final Intelligence Output report."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}

    print("\n")
    print("╔" + "═" * 76 + "╗")
    print("║" + "  🧠  INTELLIGENCE OUTPUT — FINAL ADVISORY REPORT".center(76) + "║")
    print("╚" + "═" * 76 + "╝")

    # ── Top 3 Crops ──
    print(f"\n  {'━' * 74}")
    print("  📊  TOP 3 RECOMMENDED CROPS")
    print(f"  {'━' * 74}\n")

    why_map = {s["crop_name"]: s["why"] for s in why_summaries}

    for r in fvi_top3:
        medal = medals.get(r["rank"], "  ")
        why = why_map.get(r["crop_name"], "No summary available.")
        print(f"  {medal} #{r['rank']}  {r['crop_name']}")
        print(f"       FVI Score: {r['fvi']:.2f}/10")
        print(f"       Components: ML={r['ml_score']:.1f}  Agro={r['s_farmer']}"
              f"  Mkt={r['s_trader']}  Demand={r['s_analyst']}")
        print(f"       💡 Why: {why}")
        print()

    # ── Fertilizer Advice ──
    print(f"  {'━' * 74}")
    print("  🧪  FERTILIZER RECOMMENDATIONS (from Soil Health Card)")
    print(f"  {'━' * 74}\n")

    for f in fertilizer:
        print(f"  ⚗️  {f['nutrient']}")
        print(f"     Status : {f['status']}")
        print(f"     Action : {f['action']}")
        print()

    # ── Sowing Timing ──
    print(f"  {'━' * 74}")
    print("  📅  SOWING TIMING ADVISORY (from Weather Forecast)")
    print(f"  {'━' * 74}\n")

    print(f"  🌤️  Window Status  : {sowing['window_status']}")
    print(f"  🌧️  48h Rain Expect: {sowing['expected_rain_48h_mm']}mm")
    print(f"  🌡️  Peak Temp      : {sowing['peak_temp_celsius']}°C")
    print(f"  🐛  Pest Alert     : {sowing['pest_alert']}")
    print(f"  💧  Irrigation     : {sowing['irrigation_advisory']}")

    print(f"\n  {'─' * 74}")
    print("  🕐  CROP-SPECIFIC SOWING WINDOWS")
    print(f"  {'─' * 74}")
    for crop_type, advice in sowing["crop_timing"].items():
        label = crop_type.replace("_", " ").title()
        print(f"  • {label}: {advice}")

    print(f"\n╔{'═' * 76}╗")
    print(f"║{'  ✅  REPORT COMPLETE — Advisory generated by Multi-Agent System'.center(76)}║")
    print(f"╚{'═' * 76}╝\n")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

async def main():
    # 1. Load data
    raw_data = load_json()
    bundle = get_bundle(raw_data)

    # 2. Show what we're feeding the agents
    crop_names = [c["crop_name"] for c in get_top5_crops(bundle)]
    print(f"\n🌾 Top 5 crops for analysis: {', '.join(crop_names)}\n")

    # ── PHASE 1: Initial parallel analysis ──
    print("━" * 60)
    print("  📌  PHASE 1 — INITIAL PARALLEL ANALYSIS")
    print("━" * 60)
    results = await run_all_agents(bundle)
    farmer_result, trader_result, analyst_result = results
    display_results(results)

    # ── PHASE 2: Cross-Critique Loop ──
    print("━" * 60)
    print("  📌  PHASE 2 — CROSS-CRITIQUE LOOP")
    print("━" * 60)
    print("  Farmer ↔ Trader will exchange feedback")
    print("  Analyst scores are carried forward as-is")
    final_farmer, final_trader, rounds = await run_cross_critique(
        bundle, farmer_result, trader_result,
    )
    display_consensus(final_farmer, final_trader, analyst_result, rounds)

    # ── PHASE 3: Final Viability Index ──
    print("━" * 60)
    print("  📌  PHASE 3 — FINAL VIABILITY INDEX (FVI)")
    print("━" * 60)
    fvi_results = compute_fvi(bundle, final_farmer, final_trader, analyst_result)
    display_fvi(fvi_results)

    # ── PHASE 4: Intelligence Output ──
    print("━" * 60)
    print("  📌  PHASE 4 — INTELLIGENCE OUTPUT")
    print("━" * 60)

    fvi_top3 = fvi_results[:3]
    fertilizer = _build_fertilizer_advice(bundle)
    sowing = _build_sowing_advisory(bundle)
    why_summaries = await _generate_why_summaries(
        fvi_top3, final_farmer, final_trader, analyst_result,
    )
    display_intelligence_output(fvi_top3, why_summaries, fertilizer, sowing)

    # Save all results
    output = {
        "phase_1_initial": results,
        "phase_2_cross_critique": {
            "rounds_taken": rounds,
            "final_farmernomist": final_farmer,
            "final_trader_strategist": final_trader,
            "analyst_unchanged": analyst_result,
        },
        "phase_3_fvi": fvi_results,
        "phase_4_intelligence_output": {
            "top_3_crops": fvi_top3,
            "why_summaries": why_summaries,
            "fertilizer_advice": fertilizer,
            "sowing_advisory": sowing,
        },
    }
    out_path = "agent_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"📄 Full results (Phase 1–4) saved to '{out_path}'\n")



# ──────────────────────────────────────────────
#  FastAPI Application
# ──────────────────────────────────────────────

app = FastAPI(
    title="Agricultural Advisory System",
    description="Multi-Agent Crop Advisory with real-time WebSocket debate logs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time agent debate logs.

    Clients connect here to receive JSON messages as agents analyze crops.
    Each message contains: event, sender, crop, text, score, phase, timestamp.
    """
    await manager.connect(websocket)
    try:
        # Keep the connection open; listen for client pings or commands
        while True:
            data = await websocket.receive_text()
            # Echo back as acknowledgement (clients can send 'ping')
            if data.strip().lower() == "ping":
                await websocket.send_json({"event": "pong", "timestamp": time.time()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/", response_class=FileResponse)
async def root():
    """Serve the premium terminal dashboard."""
    return FileResponse("index.html")


@app.post("/run")
async def trigger_pipeline():
    """HTTP trigger to start the full agent pipeline.

    Broadcasts phase transitions and agent logs via WebSocket
    as the pipeline executes.
    """
    await broadcast_log("System", "Pipeline started", event="pipeline_start")

    try:
        await main()
        await broadcast_log("System", "Pipeline complete", event="pipeline_end")
        return {"status": "success", "message": "Pipeline completed"}
    except Exception as e:
        await broadcast_log("System", f"Pipeline error: {e}", event="pipeline_error")
        return {"status": "error", "message": str(e)}


# ──────────────────────────────────────────────
#  Entrypoint
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🌐 Starting Agricultural Advisory Server on http://localhost:8000")
    print("   WebSocket endpoint: ws://localhost:8000/ws")
    print("   Test page:          http://localhost:8000/")
    print("   Trigger pipeline:   POST http://localhost:8000/run\n")

    uvicorn.run(
        "agents:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
