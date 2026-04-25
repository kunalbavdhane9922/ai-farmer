import json
import asyncio
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ollama import AsyncClient
from duckduckgo_search import DDGS
from data_loader import load_json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
#  WebSocket Manager
# ──────────────────────────────────────────────
class ConnectionManager:
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
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                pass

manager = ConnectionManager()

async def broadcast_log(sender: str, text: str, phase: str, event: str = "agent_log", crop: str = None):
    msg = {
        "event": event,
        "sender": sender,
        "text": text,
        "phase": phase
    }
    if crop:
        msg["crop"] = crop
    await manager.broadcast(msg)

# ──────────────────────────────────────────────
#  Constants & Internet Search
# ──────────────────────────────────────────────
MODEL = "gemma3"

RESPONSE_SCHEMA = """
{
  "agent_name": "<string>",
  "thinking_process": "<string: Your detailed, conversational response to the group discussing your findings. Provide a rich 2-3 paragraph discussion based heavily on the live internet findings and ML data.>",
  "analysis": [
    {
      "crop_name": "<string>",
      "evaluation": "<string: A 2-sentence qualitative evaluation of this crop based on your domain and internet findings. No numeric scores.>"
    }
  ]
}
""".strip()

def internet_search(query: str, max_results: int = 3) -> str:
    """Perform a web search and return a compiled string of results."""
    print(f"  🌐 [Search] Searching internet for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No recent internet data found."
            formatted = []
            for r in results:
                formatted.append(f"- {r.get('title')}: {r.get('body')}")
            return "\n".join(formatted)
    except Exception as e:
        return f"Internet search failed: {str(e)}"

# ──────────────────────────────────────────────
#  Prompt Builders with Real-Time Internet Data
# ──────────────────────────────────────────────

def _get_ml_crops(bundle: dict) -> list:
    """Extract ONLY the ML Engine crop data."""
    try:
        return bundle.get("knowledge_bundle", {}).get("ml_engine_crop_suitability", {}).get("top_5_crops", [])
    except:
        return []

def _build_agronomist_prompt(bundle: dict) -> str:
    ml_crops = _get_ml_crops(bundle)
    crop_names = [c.get("crop_name", "Unknown") for c in ml_crops]
    
    # Perform real-time internet search for weather and soil
    search_query = f"current weather forecast and agricultural soil conditions in India 2026 for {', '.join(crop_names)}"
    internet_data = internet_search(search_query)

    return f"""You are **Agent A — Farmer**, an expert in soil science and agro-meteorology.

## Shortlisted Crops for this Season
{json.dumps(ml_crops, indent=2)}

## Recent Field Reports & Weather Updates
{internet_data}

## Your Task
Analyze these crops based on the recent field reports, weather updates, and soil conditions.
Discuss your thoughts conversationally with your partners (Trader and Analyst). Do not use numeric scores. Provide a detailed qualitative evaluation.

CRITICAL INSTRUCTION: Completely immerse yourself in the character. Do NOT mention being an AI, an ML model, or using internet searches. Talk as if you are a real, experienced farmer looking at field data. Use natural, conversational language in your "thinking_process".

## Output Format (strict JSON, no extra text)
{RESPONSE_SCHEMA}

Set "agent_name" to "Farmer".
Respond ONLY with the JSON object. No markdown fences, no commentary."""


def _build_market_strategist_prompt(bundle: dict) -> str:
    ml_crops = _get_ml_crops(bundle)
    crop_names = [c.get("crop_name", "Unknown") for c in ml_crops]
    
    search_query = f"current mandi prices, commodity market trend, and logistics demand in India 2026 for {', '.join(crop_names)}"
    internet_data = internet_search(search_query)

    return f"""You are **Agent B — Trader**, an expert in agricultural commodity markets and supply-chain logistics.

## Shortlisted Crops for this Season
{json.dumps(ml_crops, indent=2)}

## Latest Market Intelligence & Price Trends
{internet_data}

## Your Task
Analyze the crops based on the latest market intelligence regarding current mandi prices, demand, and logistics.
Discuss your thoughts conversationally with your partners. Do not use numeric scores. Evaluate the marketability of each crop.

CRITICAL INSTRUCTION: Completely immerse yourself in the character. Do NOT mention being an AI, an ML model, or using internet searches. Talk as if you are a real, seasoned trader analyzing the market. Use natural, conversational language in your "thinking_process".

## Output Format (strict JSON, no extra text)
{RESPONSE_SCHEMA}

Set "agent_name" to "Trader".
Respond ONLY with the JSON object. No markdown fences, no commentary."""


def _build_trend_forecaster_prompt(bundle: dict) -> str:
    ml_crops = _get_ml_crops(bundle)
    crop_names = [c.get("crop_name", "Unknown") for c in ml_crops]
    
    search_query = f"Indian government agricultural policies, export demand, and MSP (Minimum Support Price) updates 2026 for {', '.join(crop_names)}"
    internet_data = internet_search(search_query)

    return f"""You are **Agent C — Analyst**, an expert in agricultural policy analysis, government support mechanisms, and global trade patterns.

## Shortlisted Crops for this Season
{json.dumps(ml_crops, indent=2)}

## Recent Policy Briefs & Export Trends
{internet_data}

## Your Task
Analyze the crops based on the recent policy briefs regarding export demand, MSP changes, and government subsidies.
Discuss your thoughts conversationally with your partners. Do not use numeric scores.

CRITICAL INSTRUCTION: Completely immerse yourself in the character. Do NOT mention being an AI, an ML model, or using internet searches. Talk as if you are a real, insightful analyst reviewing government policies. Use natural, conversational language in your "thinking_process".

## Output Format (strict JSON, no extra text)
{RESPONSE_SCHEMA}

Set "agent_name" to "Analyst".
Respond ONLY with the JSON object. No markdown fences, no commentary."""


# ──────────────────────────────────────────────
#  Agent Runner
# ──────────────────────────────────────────────
async def _call_agent(agent_label: str, prompt: str, *, phase: str = "") -> dict:
    client = AsyncClient()
    print(f"  🚀 [{agent_label}] Sending request to Ollama ({MODEL})…")

    # Broadcast: agent is starting (trigger typing indicator on UI)
    await broadcast_log(sender=agent_label, text="", phase=phase, event="agent_dispatch")

    t0 = time.perf_counter()
    # We use a higher temperature for more conversational and diverse output
    response = await client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.6},
        format="json",
    )
    elapsed = time.perf_counter() - t0
    raw = response["message"]["content"]
    print(f"  ✅ [{agent_label}] Response received in {elapsed:.1f}s")

    # Stop typing indicator
    await broadcast_log(sender=agent_label, text="", phase=phase, event="agent_status")

    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1)
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1)
    if cleaned.endswith("```"):
        cleaned = cleaned[::-1].replace("```", "", 1)[::-1]
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        
        # Broadcast the AI's actual original thinking as a conversational message
        thinking = result.get("thinking_process", "")
        if thinking:
            await broadcast_log(sender=agent_label, text=thinking, phase=phase, event="agent_log")
            
        return result
    except json.JSONDecodeError as e:
        print(f"  ⚠️  [{agent_label}] JSON parse failed.")
        return {"agent_name": agent_label, "analysis": []}

SKIP_REQUESTED = False

# ──────────────────────────────────────────────
#  Final Report Generator
# ──────────────────────────────────────────────
async def _generate_final_report(history: str) -> str:
    prompt = f"""Based on the following deep discussion between the Farmer, Trader, and Analyst, create a definitive final recommendation report for the user.
The report must accurately reflect the consensus (or trade-offs) discussed by the experts. Do NOT fake or hallucinate data; stick to what they discussed.

CRITICAL FORMATTING RULES:
1. Do NOT write this as a letter, email, or memo. 
2. Do NOT include "To:", "From:", "Date:", or any opening pleasantries. 
3. Go straight into the content using beautiful, structured Markdown.
4. You MUST use exactly these 5 Markdown Headings (e.g. `## 1. Recommended Crops`):

## 1. Recommended Crops
(Which crop(s) are the ultimate winners)

## 2. Profit
(Use a Markdown Table to show each crop wise yearly profit estimates or potential based on the discussion)

## 3. Risk
(Highlight any weather or market risks mentioned)

## 4. Reason
(Why the recommended crops were chosen)

## 5. Avoid
(Crops to completely avoid and why)

Discussion History:
{history}
"""
    client = AsyncClient()
    print("  🚀 [System] Generating Final Report...")
    await broadcast_log("System", "Generating Final Report based on the debate...", "final_report_generation", "report_start")
    response = await client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.4},
    )
    report = response["message"]["content"]
    await broadcast_log("System", report, "final_report", "final_report")
    return report

# ──────────────────────────────────────────────
#  Pipeline
# ──────────────────────────────────────────────
async def run_pipeline(bundle: dict):
    global SKIP_REQUESTED
    # Phase 1: Initial Research & Discussion
    await broadcast_log("System", "Phase 1: Agents are searching the internet for real-time data and formulating their thoughts...", "phase_1", "phase_start")

    # Run searches and agents concurrently
    farmer_task = asyncio.create_task(_call_agent("Farmer", _build_agronomist_prompt(bundle), phase="phase_1"))
    trader_task = asyncio.create_task(_call_agent("Trader", _build_market_strategist_prompt(bundle), phase="phase_1"))
    analyst_task = asyncio.create_task(_call_agent("Analyst", _build_trend_forecaster_prompt(bundle), phase="phase_1"))

    farmer_res, trader_res, analyst_res = await asyncio.gather(farmer_task, trader_task, analyst_task)

    # Broadcast the evaluations
    for res, name in [(farmer_res, "Farmer"), (trader_res, "Trader"), (analyst_res, "Analyst")]:
        for c in res.get("analysis", []):
            await broadcast_log(
                sender=name,
                text=c.get("evaluation", "No evaluation provided."),
                crop=c.get("crop_name", "Unknown"),
                phase="phase_1"
            )

    # Build history context
    history = f"Farmer's thoughts: {farmer_res.get('thinking_process')}\n" \
              f"Trader's thoughts: {trader_res.get('thinking_process')}\n" \
              f"Analyst's thoughts: {analyst_res.get('thinking_process')}"
    full_history = history

    if not SKIP_REQUESTED:
        # Phase 2: Cross-Critique Discussion
        await broadcast_log("System", "Phase 2: Agents are reviewing each other's internet findings and debating...", "cross_critique", "critique_round_start")

        farmer_critique_prompt = f"""You are **Farmer**. Read your partners' thoughts:\n{history}\n
Write a 2-paragraph response addressing the Trader and Analyst directly in a group chat format. How does their market/policy data change your view?
CRITICAL INSTRUCTION: Completely immerse yourself in the character. Do NOT mention being an AI, an ML model, or using internet searches. Speak naturally as a Farmer.
Format your response exactly using this JSON schema:
{RESPONSE_SCHEMA}
"""
        trader_critique_prompt = f"""You are **Trader**. Read your partners' thoughts:\n{history}\n
Write a 2-paragraph response addressing the Farmer and Analyst directly in a group chat format. How does their weather/policy data change your view?
CRITICAL INSTRUCTION: Completely immerse yourself in the character. Do NOT mention being an AI, an ML model, or using internet searches. Speak naturally as a Trader.
Format your response exactly using this JSON schema:
{RESPONSE_SCHEMA}
"""
        
        fc_task = asyncio.create_task(_call_agent("Farmer", farmer_critique_prompt, phase="cross_critique"))
        tc_task = asyncio.create_task(_call_agent("Trader", trader_critique_prompt, phase="cross_critique"))
        
        farmer_crit_res, trader_crit_res = await asyncio.gather(fc_task, tc_task)
        
        for res, name in [(farmer_crit_res, "Farmer"), (trader_crit_res, "Trader")]:
            for c in res.get("analysis", []):
                await broadcast_log(
                    sender=name,
                    text=c.get("evaluation", "No evaluation provided."),
                    crop=c.get("crop_name", "Unknown"),
                    phase="cross_critique"
                )
        full_history += f"\nFarmer's Critique: {farmer_crit_res.get('thinking_process')}\nTrader's Critique: {trader_crit_res.get('thinking_process')}"
    else:
        await broadcast_log("System", "User skipped Phase 2. Moving straight to Final Report generation...", "cross_critique", "system_log")

    # Phase 3: Final Report
    await _generate_final_report(full_history)

    await broadcast_log("System", "Debate concluded.", "done", "pipeline_end")
    return {"status": "success"}

# ──────────────────────────────────────────────
#  FastAPI Endpoints
# ──────────────────────────────────────────────
@app.get("/")
def serve_dashboard():
    return FileResponse("index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global SKIP_REQUESTED
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                parsed = json.loads(data)
                if parsed.get("action") == "skip":
                    SKIP_REQUESTED = True
                    print("  ⏭️ [System] Skip requested by user.")
            except:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/run")
async def trigger_pipeline():
    global SKIP_REQUESTED
    SKIP_REQUESTED = False
    try:
        bundle = load_json(Path("all_agent_data.json"))
    except Exception as e:
        return {"error": str(e)}

    # Run in background so HTTP response returns immediately
    asyncio.create_task(run_pipeline(bundle))
    return {"message": "Pipeline started"}

if __name__ == "__main__":
    import uvicorn
    print("\n🌐 Starting Agricultural Advisory Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
